use std::io::{self, BufReader, BufRead, Write};
use std::fs::File;
use std::ops::Add;

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
    fn resolve_boundary_condition(&mut self, next: &mut Coordinate, bound: &Coordinate) -> f64;
    fn acceleration(&self, intermediate_velocity: &Coordinate) -> Coordinate;
}
impl NumericTools for Object {
    fn ground_linear_interpolate(&self, next: &Coordinate) -> Coordinate {
        let x = self.pos.x + (0.0 - self.pos.y) * (next.x - self.pos.x) / (next.y - self.pos.y);
        Coordinate { x, y: 0.0 }
    }
    fn resolve_boundary_condition(&mut self, next: &mut Coordinate, bound: &Coordinate) -> f64 {
        // Determine the direction that the object is potentially colliding from 
        // Check if self and next are both to right or left of boundary condition


        if (self.pos.x - bound.x) * (next.x - bound.x) > 0.0 {
            return self.vel.x
        }

        let real_bound = if self.pos.x < bound.x {
            bound.x - self.tol 
        } else {
            bound.x + self.tol
        };

        // Find interpolated y value through barrier
        let ratio = (real_bound - self.pos.x) / (next.x - self.pos.x);
        let y_i =  self.pos.y + (next.y - self.pos.y) / ratio;

        // println!("{:?}", self.vel);
        // Check if value clears the barrier
        if y_i > bound.y + self.tol {
            return self.vel.x
        }
        // println!("Between y_i {} and y_i+1 {} Interpolated y_int {y_i}", self.pos.y, next.y);

        // Return interpolated coordinate 
        *next = Coordinate { x: real_bound, y: y_i };
        return self.vel.x * -1.0;
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
    fn runge_kutta(&mut self, angle: &f64) -> Vec<Coordinate>;
    fn euler(&mut self, angle: &f64) -> Vec<Coordinate>;
}
impl NumericMethods for Object {
    fn runge_kutta(self: &mut Object, angle: &f64) -> Vec<Coordinate> {
        // Clear previous position and velocity values
        self.pos.clear(); 
        self.vel.resolve(&self.v_i, angle); 
        
        // Make copy of boundary conditions 
        let screen_bound = Coordinate { x: self.d_s, y: self.h_s };
        let wall_bound   = Coordinate { x: self.d_w, y: self.h_w };

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
            // println!("{}", next.x - self.pos.x);
            self.vel = self.vel + accel_weighted_sum.scale(&self.dt);

            // Check for screen collision 
            self.vel.x = self.resolve_boundary_condition(&mut next, &screen_bound); 
            //println!("Velocity: {:?}", self.vel); 
            // Check for wall collision 
            self.vel.x = self.resolve_boundary_condition(&mut next, &wall_bound);

            // Check if intersecting with ground; if so rectify through interpolation and return
            if next.y <= self.tol && self.vel.y < 0.0 {
                next = self.ground_linear_interpolate(&next); 
                positive = false;
            }
            // println!("Position {:?}", next);
            // Add position to vector 
            self.pos = next;
            positions.push(self.pos)
        }
        positions
    }
    fn euler(&mut self, angle: &f64) -> Vec<Coordinate> {
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
    }
}

struct Trajectory {
    object: Object,
    method: fn(&mut Object, &f64) -> Vec<Coordinate> 
}

impl Trajectory {
    fn new(object: Object, method: fn(&mut Object, &f64) -> Vec<Coordinate> ) -> Trajectory {
        Trajectory {
           object, 
           method
        }   
    }
}


impl Trajectory {
    // Increase bounds by one until valid
    fn set_bounds(&mut self, angle_bounds: &mut Coordinate, bounds: &mut Coordinate) {
        bounds.x = (self.method)(&mut self.object, &angle_bounds.x).last().unwrap().x;
        bounds.y = (self.method)(&mut self.object, &angle_bounds.y).last().unwrap().x;

        println!("Bounds [{}, {}]", bounds.x, bounds.y);

        // While the bounds lie on the same side of the target (i.e. not bounding solution)
        while (bounds.x - self.object.targ) * (bounds.y - self.object.targ) > 0.0 {
            angle_bounds.y += 1.0_f64.to_radians();
            bounds.y = (self.method)(&mut self.object, &angle_bounds.y).last().unwrap().x;
            // println!("Angle {}: Boundary Error: {}", angle_bounds.y, bounds.y - self.object.targ);
        } 
    }
}

trait NumericSolution {
    fn bisection(&mut self, angle_bounds: &mut Coordinate) -> (Vec<f64>, f64);
    //fn secant() -> (Vec<f64>, f64);
}
// Both methods return the error vector 
impl NumericSolution for Trajectory {
    fn bisection(&mut self, angle_bounds: &mut Coordinate) -> (Vec<f64>, f64) {
        // Copy angles locally 
        let mut error: Vec<f64> = Vec::new();
        let mut i = 0;
        let mut midpoint: f64 = 0.0;

        let mut searching = true;
        while searching == true && i < 50 {
            midpoint = angle_bounds.average(); 
            let x_final = (self.method)(&mut self.object, &midpoint).last().unwrap().x;
            let current_error = x_final - self.object.targ;
            error.push(current_error);
    
            if current_error.abs() < self.object.tol {
                searching = false;
            } else if current_error > 0.0 {
                angle_bounds.y = midpoint; 
            } else {
                angle_bounds.x = midpoint;
            }

            println!("Iteration {i}; Error: {}; Angle: {}", error[i], midpoint.to_degrees());
            i += 1;
        }
        
        (error, midpoint)
    }
    /*fn secant() -> Vec<f64> {
    
    }*/
}

fn main() -> io::Result<()> {
    let input = "inputs1-3.txt".to_string();
    let object: Object = Object::new(input)?;
    println!("{:?}", object);
    let mut calculator = Trajectory::new(object, Object::euler);

    let mut angle_bounds = Coordinate { x: 0.0_f64.to_radians(), y: 25.0_f64.to_radians() };
    let mut bounds = Coordinate { x: 0.0, y: 0.0 };

    // Compute angular bounds for problem
    calculator.set_bounds(&mut angle_bounds, &mut bounds);

    println!("Angle Bounds: {:?}", angle_bounds);
    println!("Physical Bounds: {:?}", bounds);

    let (_, optimal_angle) = calculator.bisection(&mut angle_bounds);
    
    println!("Optimal Angle {}", optimal_angle.to_degrees());
    let results = calculator.object.euler(&optimal_angle.to_radians());

    let mut file = File::create("optimal_angle.txt")?;
    for value in results {
        writeln!(file, "{:?}", value)?;
    } 

    Ok(())
}
