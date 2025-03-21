// Example functions to integrate over 
pub fn square(x: f64) -> f64 {
    x * x
}

pub fn sin_cubed(x: f64) -> f64 {
    (x.sin()).powf(3.0)
}

// Factorial function to create cache at compile time 
const fn cache_factorial(n: usize) -> u64 {
    // Sets base case to 1 else computes nth 
    match n {
        0 => 1, 
        _ => {
            // Initializes to 1
            let mut result: u64 = 1;
            let mut i = 1;
            // Multiplies result to each subsequent i 
            while i <= n {
                result *= i as u64;
                i += 1;
            }
            result 
        }
    }
}

// Create cache at compile time  
const CACHESIZE: usize = 21;
const CACHE: [u64; CACHESIZE] = {
    // Create array slice at compile time 
    let mut cache = [0; CACHESIZE];
    let mut i = 0;
    // Fill each index with the factorial(index)
    while i < CACHESIZE {
        cache[i] = cache_factorial(i);
        i += 1;
    }
    cache
};

// Polynomial defined by its vector coefficients and degree
pub struct Polynomial {
    coefficients: Vec::<f64>,
    degree: u64
}

// Function struct contains defined structure and its degree 
#[derive(Clone, Copy)]
pub struct Function {
    func: fn(f64) -> f64,
    degree: u64         // If a function does not have a degree this refers to node count    
}

// Mixed struct of a summation of functions with corresponding coefficients
pub struct SeriesFunction {
    funcs: Vec<Function>,
    func_exp: Vec<u64>,     // Defines the exponents per func -> taylor series expansions 
    poly: Polynomial        // Coefficients associated with each subfunction f_i. Also contains degree
}


// Evaluate trait implemenation 
pub trait Evaluate {
    fn evaluate(&self, x: f64) -> f64;      // Evaluate the function/polynomial/etc at a point
    fn degree(&self) -> u64;                // Fetch Degree/Node count for f/p/etc.
}
impl Evaluate for Function {
    // Call function to get evaluate at point 
    fn evaluate(&self, x: f64) -> f64 {
        (self.func)(x)
    }
    // Pull degree
    fn degree(&self) -> u64 {
        self.degree
    }
}
impl Evaluate for Polynomial {
    // Evaluate polynomial based on coefficients at point 
    fn evaluate(&self, x: f64) -> f64 {
        let mut result: f64 = 0.0;

        // Degree of each coefficient corresponds to ith index
        for (i, &coeff) in self.coefficients.iter().enumerate() {
           result += coeff * x.powi(i as i32); 
        }
        result
    }
    // Pull Degree
    fn degree(&self) -> u64 {
        self.degree
    }
}
// Implementations of Evaluate and Integrate on a SeriesFunction 
impl Evaluate for SeriesFunction {
    // Evaluate a series polynomial 
    fn evaluate(&self, x: f64) -> f64 {
        let mut result: f64 = 0.0;

        // Evaluate the ith function associated with the ith coefficient at x and sum
        for (i, &coeff) in self.poly.coefficients.iter().enumerate() {
            let eval = self.funcs[i].evaluate(x);
            result += coeff * eval.powf(self.func_exp[i] as f64);
        }
        result     
    }
    fn degree(&self) -> u64 {
        self.poly.degree
    }
}

// Takes any structure that implements Evaluate (which insinuates having a member for the degree)
pub trait Integrate {
    fn integrate<F>(&self, a: f64, b: f64, func: F) -> f64
    where 
        F: Fn(&dyn Evaluate, f64, f64) -> f64;      // Creates wide pointer to struct
}

impl Integrate for Polynomial {
    fn integrate<F>(&self, a: f64, b: f64, func: F) -> f64
    where 
        F: Fn(&dyn Evaluate, f64, f64) -> f64,
    {
        func(self, a, b)
    }
}
impl Integrate for Function {
    fn integrate<F>(&self, a: f64, b: f64, func: F) -> f64 
    where 
        F: Fn(&dyn Evaluate, f64, f64) -> f64,
    {
        func(self, a, b)
    }
}
impl Integrate for SeriesFunction {
    fn integrate<F>(&self, a: f64, b: f64, func: F) -> f64 
    where
        F: Fn(&dyn Evaluate, f64, f64) -> f64,
    {
        func(self, a, b)
    }
}               // Implementing integrate is identical for structs persuant to &dyn Evaluate
 
// Simple numeric trapezoid integration for input function on interval
pub fn trapezoid(func: &dyn Evaluate, a: f64, b: f64) -> f64 {
    let mut scale_factor = (1.0 / (func.degree() as f64)).ln();
    if scale_factor < 1.0 {
        scale_factor = 1.0;
    }
    let n = 100000 * scale_factor as u64;       // Computes node count dynamically from degree 

    let height: f64 = (b - a) / (n as f64);     // Find height of trapezoid
    let mut integral: f64 = 0.0;

    // Sets initial value to be endpoints
    integral += 0.5 * func.evaluate(b) + 0.5 * func.evaluate(a);

    // Loops over all trapezoids
    for i in 0..n {
        let dx: f64 = a + (i as f64) * height; 
        integral += func.evaluate(dx);
    }

    // Scales by height and returns
    integral *= height;
    integral
}

// Polynomial "Constructor". Generates a polynomial of requested size 
pub fn generate_polynomial(in_degree: u64) -> Polynomial {
    let mut in_coefficients: Vec<f64> = Vec::with_capacity((in_degree + 1) as usize);
    // Sets the coefficients to random values
    for _ in 0..in_degree {
        let coeff = rand::random_range(0.0..=5.0);
        in_coefficients.push(coeff);
    }

    // Returns polynomial
    Polynomial {
        coefficients: in_coefficients,
        degree: in_degree,
    }
}

fn main() {
    let upper = rand::random_range(5..=10);
    let lower = rand::random_range(1..=5);

    // Example of a function 
    let square_fn = Function {
        func: square, 
        degree: 2
    };
    
    // Example of a random polynomial 
    let poly = generate_polynomial(5);    

    println!("Lower {} and Upper {} bounds.", lower, upper);

    // As an example lets define a series function that is a 5th order taylor series expansion for e^x
    let identity = Function {
        func: |x: f64| x,
        degree: 1
    };

    let init_degree: u64 = 20;
    let terms: u64 = init_degree + 1;
    let mut set_coefficients: Vec::<f64> = Vec::with_capacity(terms as usize);
    let mut set_exponents: Vec::<u64> = Vec::with_capacity(terms as usize);

    for i in 0..=init_degree {
         set_coefficients.push(1.0 / (CACHE[i as usize] as f64));
         set_exponents.push(i);
    }

    // println!("Coefficients: {:?} Exponents {:?}", set_coefficients, set_exponents);
    
    let e_x = SeriesFunction {
        funcs: vec![identity; terms as usize],
        func_exp: set_exponents,
        poly: Polynomial {
            coefficients: set_coefficients,
            degree: init_degree
        },
    };

    // println!("Polynomial: {:?}", poly.coefficients);

    // Pass specific function pointers to trapezoid integration calculation
    let square_result = square_fn.integrate(lower as f64, upper as f64, trapezoid);
    let poly_result = poly.integrate(lower as f64, upper as f64, trapezoid); 
    let e_x_result  = e_x.integrate(lower as f64, upper as f64, trapezoid);
    let exp_func = Function {
        func: |x: f64| x.exp(),
        degree: init_degree
    };
    let e_x_eval = e_x.evaluate(3.0);

    // Print Results 
    println!("Square Integral Solution: {}", square_result);
    println!("Polynomial Solution: {}", poly_result);
    println!("{init_degree}th Order e^x Approximation at x=3 {}", e_x_eval);
    println!("Approximate Integration Solution {}", e_x_result);
    println!("Error for Integration: {}", exp_func.integrate(lower as f64, upper as f64, trapezoid) - e_x_result);
}
