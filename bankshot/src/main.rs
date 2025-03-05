use std::io::{self, BufReader, BufRead, Write};
use std::fs::File;
use std::ops::{Add, Sub};

#[derive(Debug, Copy, Clone, PartialEq)]
struct Coordinate {
    x: f64,
    y: f64,
}

impl Add for Coordinate {
    type Output = Self;

    fn add(self, other: Coordinate) -> Self {
        Coordinate {
            x: self.x + other.x,
            y: self.y + other.y
        }
    }
}
impl Sub for Coordinate {
    type Output = Self;

    fn sub(self, other: Coordinate) -> Self {
        Coordinate {
            x: self.x - other.x,
            y: self.y - other.y,
        } 
    }
}

impl Coordinate {
    fn clear(&mut self) {
        self.x = 0.0;
        self.y = 0.0
    }
    fn resolve(&mut self, magnitude: &f64, angle: &f64) {
        self.x = magnitude * angle.cos();
        self.y = magnitude * angle.sin()
    }
    fn magnitude(&self) -> f64 {
        (self.x * self.x + self.y * self.y).sqrt()
    }
    fn mut_scale(&mut self, scalar: &f64) {
        self.x *= scalar; 
        self.y *= scalar; 
    }
    fn scale(self, scalar: &f64) -> Self {
        Coordinate {
            x: self.x * scalar,
            y: self.y * scalar,
        }
    }
    fn average(self) -> f64 {
        (self.x + self.y) / 2.0
    }
    fn to_degrees(self) -> Self {
        Coordinate {
            x: self.x.to_degrees(),
            y: self.y.to_degrees()
        }
    }
}

#[derive(Debug, Copy, Clone)]
struct Object {
    pos: Coordinate,
    vel: Coordinate,
    mass: f64,
    drag: f64, 
    v_i:  f64,
    d_s:  f64,
    h_s:  f64,
    targ: f64,
    d_w:  f64, 
    h_w:  f64,
    w_v:  f64,
    dt:   f64, 
    tol:  f64
}

// Returns new Object for an inputs file or io error for invalid file
impl Object {
    fn new(path: String) -> Result<Object, io::Error> {

        let file: File = File::open(path)?;
        let reader     = BufReader::new(file); 

        let mut lines = reader.lines();
        
        // Define closure to parse a line when called
        let mut parse_member = || -> Result<f64, io::Error> {
            lines.next()
                .ok_or(io::Error::new(
                    io::ErrorKind::UnexpectedEof,
                    "Invalid File Structure"
                ))?
                .and_then(|line| {
                    line.trim()
                        .parse::<f64>()
                        .map_err(|e| io::Error::new(io::ErrorKind::InvalidData, e))
                })
        }; 

        // Return a valid object if parse member local function works as expected
        Ok(Object {
            pos: Coordinate { x: 0.0, y: 0.0 },
            vel: Coordinate { x: 0.0, y: 0.0 },
            mass: parse_member()?,
            drag: parse_member()?, 
            v_i:  parse_member()?,
            d_s:  parse_member()?,
            h_s:  parse_member()?,
            targ: parse_member()?,
            d_w:  parse_member()?, 
            h_w:  parse_member()?,
            w_v:  parse_member()?,
            dt:   parse_member()?, 
            tol:  parse_member()?
        })
    }
} 

trait NumericTools {
    fn ground_linear_interpolate(&self, coordinate: &Coordinate) -> Coordinate;
    fn resolve_boundary_condition(&mut self, next: &mut Coordinate, next_vel: &mut Coordinate, bound: &Coordinate) -> bool;
    fn acceleration(&self, intermediate_velocity: &Coordinate) -> Coordinate;
}
impl NumericTools for Object {
    fn ground_linear_interpolate(&self, next: &Coordinate) -> Coordinate {
        let x = self.pos.x + (0.0 - self.pos.y) * (next.x - self.pos.x) / (next.y - self.pos.y);
        Coordinate { x, y: 0.0 }
    }
    fn resolve_boundary_condition(&mut self, next: &mut Coordinate, next_vel: &mut Coordinate, bound: &Coordinate) -> bool {
        // Determine the direction that the object is potentially colliding from 
        // Check if self and next are both to right or left of boundary condition

        if (self.pos.x - bound.x) * (next.x - bound.x) >= 0.0 {
            return false 
        }

        // Find interpolated y value through barrier
        let ratio = (bound.x - self.pos.x) / (next.x - self.pos.x);
        let y_i =  self.pos.y + (next.y - self.pos.y) / ratio;

        // Check if value clears the barrier
        if y_i > bound.y + self.tol {
            return false
        }

        let dt_i = ratio * self.dt;

        // Return interpolated coordinate 
        *next = Coordinate { x: bound.x, y: y_i };

        let a = self.acceleration(next_vel);
        *next_vel = *next_vel + a.scale(&dt_i);
        next_vel.x *= -1.0;

        return true
        // println!(">> Collision!\n>> Collision!\n {:?}", self.vel);
    }
    fn acceleration(&self, intermediate_velocity: &Coordinate) -> Coordinate {
        // Find effective velocity
        let velo = Coordinate { x: intermediate_velocity.x - self.w_v, y: intermediate_velocity.y }; 
        let effective_velo = velo.magnitude();

        // Find acceleration components 
        let ax = -self.drag * velo.x * effective_velo;
        let ay = -self.mass * 9.81 - self.drag * velo.y * effective_velo;

        // Create new acceleration coordiante pair
        let mut acceleration = Coordinate { x: ax, y: ay };

        // Scale force by mass and return acceleration binding 
        acceleration.mut_scale(&(1.0 / self.mass));
        acceleration
    }
}

trait NumericMethods {
    fn runge_kutta(&mut self, angle: &f64, screen: bool) -> (Vec<Coordinate>, bool, bool);
    //fn euler(&mut self, angle: &f64) -> Vec<Coordinate>;
}
impl NumericMethods for Object {
    fn runge_kutta(self: &mut Object, angle: &f64, screen: bool) -> (Vec<Coordinate>, bool, bool) {
        // Clear previous position and velocity values
        self.pos.clear(); 
        self.vel.resolve(&self.v_i, angle); 
        
        // Make copy of boundary conditions 
        let screen_bound = Coordinate { x: self.d_s, y: self.h_s };
        let wall_bound   = Coordinate { x: self.d_w, y: self.h_w };
        let mut _display = true;
        let mut hit = false;

        // Set position vector 
        let mut positive = true;
        let mut positions: Vec<Coordinate> = Vec::new(); 
        while positive {

            // Borrow velocity for all weighted steps 
            // Find weighted k values 
            let k1_velocity = &self.vel;
            let k1_accel    = self.acceleration(k1_velocity);

            let k2_velocity = &(self.vel + k1_accel.scale(&(0.5 * self.dt)));
            let k2_accel    = self.acceleration(k2_velocity);

            let k3_velocity = &(self.vel + k2_accel.scale(&(0.5 * self.dt)));
            let k3_accel    = self.acceleration(k3_velocity);

            let k4_velocity = &(self.vel + k3_accel.scale(&self.dt));
            let k4_accel    = self.acceleration(k4_velocity);

            // Sum weighted k values
            let velocity_weighted_sum = (*k1_velocity + k2_velocity.scale(&2.0) + k3_velocity.scale(&2.0) + *k4_velocity)
                .scale(&(1.0/6.0));

            let accel_weighted_sum = (k1_accel + k2_accel.scale(&2.0) + k3_accel.scale(&2.0) + k4_accel)
                .scale(&(1.0/6.0));

            // Iterate position and velocity
            let mut next = self.pos + velocity_weighted_sum.scale(&self.dt);

            /*if hit {
                println!("Current Pos:   {:?}", self.pos);
                println!("Next Position: {:?}", next);
            }*/

            // println!("{}", next.x - self.pos.x);
            let mut next_vel = self.vel + accel_weighted_sum.scale(&self.dt);

            // Check for screen collision 
            if screen == true {
                let screen_hit = self.resolve_boundary_condition(&mut next, &mut next_vel, &screen_bound); 
                if screen_hit { 
                    return (positions, hit, true)
                }
            }
                /*if hit && display {
                println!("Velocity: {:?}", self.vel);
                display = false;
            }*/
            //println!("Velocity: {:?}", self.vel); 
            // Check for wall collision 
            let local_hit = self.resolve_boundary_condition(&mut next, &mut next_vel, &wall_bound);
            if local_hit {
                //println!("HIT BACK WALL");
                hit = true;
            }

            // Check if intersecting with ground; if so rectify through interpolation and return
            if next.y <= self.tol && self.vel.y < 0.0 {
                next = self.ground_linear_interpolate(&next); 
                positive = false;
            }
            // println!("Position {:?}", next);
            // Add position to vector 
            self.pos = next;
            self.vel = next_vel;
            positions.push(self.pos)
        }
        // println!("{:?}", hit);
        (positions, hit, false)
    }
    /*fn euler(&mut self, angle: &f64) -> Vec<Coordinate> {
        // Clear pos and get velocity for angle
        self.pos.clear(); 
        self.vel.resolve(&self.v_i, angle); 
        
        // Set bounds
        let screen_bound = Coordinate { x: self.d_s, y: self.h_s };
        let wall_bound   = Coordinate { x: self.d_w, y: self.h_w };

        // Init loop
        let mut positive = true;
        let mut positions: Vec<Coordinate> = Vec::new(); 
        while positive {

            // Updates based on euler's method 
            let mut next = self.pos + self.vel.scale(&self.dt);
            self.vel = self.vel + self.acceleration(&self.vel).scale(&self.dt);

            // Check for screen collision 
            self.resolve_boundary_condition(&mut next, &screen_bound); 
            // Check for wall collision 
            self.resolve_boundary_condition(&mut next, &wall_bound);

            // Standard check for hitting ground
            if next.y <= self.tol && self.vel.y < 0.0 {
                next = self.ground_linear_interpolate(&next);
                positive = false;
            }
            self.pos = next;
            positions.push(self.pos)
        }
        positions
    }*/
}

struct Trajectory {
    object: Object,
    method: fn(&mut Object, &f64, bool) -> (Vec<Coordinate>, bool, bool)
}

impl Trajectory {
    fn new(object: Object, method: fn(&mut Object, &f64, bool) -> (Vec<Coordinate>, bool, bool) ) -> Trajectory {
        Trajectory {
           object, 
           method
        }   
    }
}


impl Trajectory {
    fn set_bounds(&mut self, angle_bounds: &mut Coordinate, bounds: &mut Coordinate) {
        let (result, _, _) = (self.method)(&mut self.object, &angle_bounds.x, false);
        bounds.x = result.last().unwrap().x;

        let (result, _, _) = (self.method)(&mut self.object, &angle_bounds.y, false);
        bounds.y = result.last().unwrap().x;

        while (bounds.x - self.object.targ) * (bounds.y - self.object.targ) > 0.0 {
            angle_bounds.y += 1.0_f64.to_radians(); 
            let (result, _, _) = (self.method)(&mut self.object, &angle_bounds.y, false);
            bounds.y = result.last().unwrap().x;

            if angle_bounds.y > 180.0_f64.to_radians() {
                break;
            }
        }
        let deg_angle = angle_bounds.y.to_degrees().ceil();
        angle_bounds.y = deg_angle.to_radians();
    }
}

trait NumericSolution {
    fn bisection(&mut self, angle_bounds: &mut Coordinate) -> (Vec<Coordinate>, f64, bool);
}
// Both methods return the error vector 
impl NumericSolution for Trajectory {
    fn bisection(&mut self, angle_bounds: &mut Coordinate) -> (Vec<Coordinate>, f64, bool) {
        // Copy angles locally 
        let mut error: Vec<Coordinate> = Vec::new();
        let mut i = 0;
        let mut midpoint: f64 = 0.0;

        let mut searching = true;
        while searching == true && i < 30 {
            midpoint = angle_bounds.average(); 
            let (result, hit_wall, _) = (self.method)(&mut self.object, &midpoint, false);
            let x_final = result.last().unwrap().x;

            let current_error = x_final - self.object.targ;
            error.push(Coordinate { x: current_error, y: midpoint });
    
            /*
            Increasing angle overall decreases deflected angle 
            A deflected angle that is < 0.0 the angle is too high so the overall
            angle should be lowered

            Likewise a deflected angle that is > 0.0 means the defl angle is too low.
            Therefore the overall should be lowered
            */            
            if current_error.abs() < self.object.tol {
                searching = false;
            } else if midpoint > 45.0 {
                if hit_wall {
                    if current_error < 0.0 {
                        angle_bounds.y = midpoint;
                    } else {
                        angle_bounds.x = midpoint;
                    }
                } else { 
                    if current_error > 0.0 {
                        angle_bounds.y = midpoint; 
                    } else {
                        angle_bounds.x = midpoint;
                    }
                }
            } else {
                if hit_wall {
                    if current_error < 0.0 {
                        angle_bounds.y = midpoint;
                    } else {
                        angle_bounds.x = midpoint;
                    }
                } else { 
                    if current_error < 0.0 {
                        angle_bounds.y = midpoint; 
                    } else {
                        angle_bounds.x = midpoint;
                    }
                }
            }

            //println!("Iteration {i}; Error: {}; Angle: {}", error[i], midpoint.to_degrees());
            i += 1;
            if i == 20 {
                return (error, midpoint, false)
            }
        }
        
        (error, midpoint, true)
    }
}

fn main() -> io::Result<()> {
    let input = "inputs1-3.txt".to_string();
    let object: Object = Object::new(input)?;
    println!("{:?}", object);
    let mut calculator = Trajectory::new(object, Object::runge_kutta);
    let mut solution_set: Vec<f64> = Vec::new();

    // Set initial bounds
    let mut angle_bounds = Coordinate { x: 0.0_f64.to_radians(), y: 0.0_f64.to_radians() };
    let mut bounds = Coordinate { x: 0.0, y: 0.0 };

    // println!("Max Angle: {}", max_angle.to_degrees());

    // Fan over 0 to 180 to find all valid angles 
    let mut last = false;
    loop {

        // Set y bound greater than x bound
        angle_bounds.y = (angle_bounds.x.to_degrees() + 2.0).to_radians();
        calculator.set_bounds(&mut angle_bounds, &mut bounds);

        println!("Angles: {:?}", angle_bounds.to_degrees());
        
        if angle_bounds.y > 180.0_f64.to_radians() {
            angle_bounds.y = 180.0_f64.to_radians();
            last = true;
        }

        let original_bounds: Coordinate = angle_bounds;
        let (errors, optimal_angle, solution) = calculator.bisection(&mut angle_bounds);
        if solution {
            println!("Bounds: {:?}", bounds);
            solution_set.push(optimal_angle.to_degrees());
        }
        // Print error regardless as a test
        for value in errors {
            print!("Error: {}, ", value.x);
            println!("Angle: {}", value.y.to_degrees());
        }
        angle_bounds.x = original_bounds.y;
        if last {
            break;
        }
    }

    // Force bounds within known successful range as a test
    let mut angle_bounds = Coordinate { x: 0.0, y: 20.0_f64.to_radians() };

    let (_, test, solution) = calculator.bisection(&mut angle_bounds);
    if solution { println!("Test Angle: {}", test.to_degrees()); }

    let (result, _, _) = calculator.object.runge_kutta(&(21.004848480224613_f64.to_radians()), false);
    let x_final = result.last().unwrap().x;

    println!("X from 14.898 deg: {x_final}");
    println!("Solution Set: {:?}", solution_set);
    
    let mut file = File::create("optimal_angle.txt")?;
    for value in result {
        write!(file, "{},", value.x)?;
        writeln!(file, "{}", value.y)?;
    } 

    Ok(())
}
