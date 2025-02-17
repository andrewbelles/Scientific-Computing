use rand::Rng;

// Example function to integrate over 
pub fn square(x: f64) -> f64 {
    return x * x;
}

pub fn sin_cubed(x: f64) -> f64 {
    return (x.sin()).powf(3.0);
}
 
// Simple numeric trapezoid integration for input function on interval
// Takes iteration count, bounds as a tuple, and a function pointer to integrand 
pub fn trapezoid_integration(n: f64, (upper_bound, lower_bound): (f64, f64), func: fn(f64) -> f64) -> f64 {
    let height: f64 = (upper_bound - lower_bound) / n;  // Find height of trapezoid
    let mut integral: f64 = 0.0;

    // Sets initial value to be endpoints
    integral += 0.5 * func(upper_bound) + 0.5 * func(lower_bound);

    // Loops over all trapezoids
    for i in 0..(n as u64) {
        let dx: f64 = lower_bound + (i as f64) * height; 
        integral += func(dx);
    }

    // Scales by height and returns
    integral *= height;
    return integral;
}

fn main() {
    let upper = rand::thread_rng().gen_range(15..=30);
    let lower = rand::thread_rng().gen_range(1..=5);

    println!("Lower {} and Upper {} bounds.", lower, upper);

    // Pass specific function pointers to trapezoid integration calculation
    let square_result = trapezoid_integration(100000.0, (upper as f64, lower as f64), square);
    let sin_cubed_result = trapezoid_integration(100000.0, (upper as f64, lower as f64), sin_cubed);

    // Print Results 
    println!("Square Integral Solution: {}", square_result);
    println!("Sin Cubed Integral Solution: {}", sin_cubed_result);
}
