use std::fmt;
use std::iter;

// Polynomial class 
#[derive(Debug, Clone)]
struct Polynomial<T> {
    coeff: Vec<T>
}

impl<T: Clone> Default for Polynomial<T> {
    fn default() -> Self {
         Polynomial { coeff: Vec::new() }
    }
}

impl<T: Clone, const N: usize> From<[T; N]> for Polynomial<T> {
    fn from(array: [T; N]) -> Self {
        let data = array.iter().cloned().collect();
        Polynomial { coeff: data }
    }
}

impl<T: fmt::Display> fmt::Display for Polynomial<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let func_str = self.coeff.iter().enumerate()
            .map(|(i, c)| {
                if i == 0 {
                    format!("{}", c)
                } else {
                    format!("{}x^{}", c, i)
                }
            })
            .collect::<Vec<_>>()
            .join(" + ");
        write!(f, "{}", func_str)
    }
}

impl<T> Polynomial<T>
where 
    T: Copy
    + fmt::Debug
    + fmt::Display
    + iter::Sum
    + num_traits::Zero
    + num_traits::Float
    + num_traits::NumCast
{
    // Trim trailing zeros then zero out values that are within tolerance 
    fn trim(&mut self) {
        let tol: T = num_traits::NumCast::from(1e-3).unwrap(); 
        while let Some(last) = self.coeff.last() {
            if last.abs() < tol {
                self.coeff.pop();
            } else {
                break
            }
        }

        self.coeff.iter_mut()
            .for_each(|c| {
                if c.abs() < tol {
                    *c = T::zero();
                }
            })
    }
    // Evaluate the polynomial at some x value 
    fn evaluate(&self, x: &T) -> T {
        // Multiplies each coefficient by the desired x coordinate to respective power i 
        self.coeff.iter().enumerate()
            .map(|(i, &c)| c * x.powi(i as i32))
            .sum()
    }
    // Return vector of roots to polynomial 
    fn get_roots(&self) -> Vec<T> {
        // Get constants to match T 
        let tol: T = num_traits::NumCast::from(1e-3).unwrap(); 
        let max_iter: u32 = 1000;
        let dx: T  = num_traits::NumCast::from(1e-2).unwrap();


        // Helper function to get derivative of function
        let derivative = |coeffs: &[T]| -> Vec<T> {
            // Applies power rule to each coefficient and down shifts such that matrix is 1 small
            coeffs.iter().enumerate()
                .skip(1)        // Skip constant value = 0 
                .map(|(i, &c)| {
                    let ti: T = num_traits::NumCast::from(i).unwrap();
                    c * ti
                })
            .collect()
        };

        // Helper function to perform synthetic division with a real valued root 
        let poly_divide = |coeffs: &[T], root: T| -> Vec<T> { 
            let degree = coeffs.len() -  1;
            let mut res: Vec<T> = Vec::with_capacity(degree);
            let mut carry = coeffs[degree];

            res.push(carry);
            // Divides out polynomial
            for i in (1..degree).rev() {
                carry = coeffs[i] + root * carry; 
                res.push(carry);
            }
            res.reverse();
            res // Ignore remainder -> Assume 0 since dividing root off  
        };

        // Returns a tuple of good lower and upper bounds 
        let bracket = |poly: &Polynomial<T>| -> Option<(T, T)> {
            let x0: T = T::zero();
            let f0: T = poly.evaluate(&x0);

            // Naive bracketing loop for set number of iterations 
            for i in 1..max_iter {
                let k: T = num_traits::NumCast::from(i).unwrap();
                let a: T = x0 - k * dx;   
                let b: T = x0 + k * dx;

                let fa: T = poly.evaluate(&a);
                let fb: T = poly.evaluate(&b);

                // On off chance we find a root return it as a tuple 
                if fa == T::zero() {
                    return Some((a, a))
                } else if fb == T::zero() {
                    return Some((b, b))
                }

                // Check for difference in sign and return 
                if fa * f0 < T::zero() {
                    return Some((a, x0))
                } else if fb * f0 < T::zero() {
                    return Some((x0, b))
                } else if fa * fb < T::zero() {
                    return Some((a, b))
                }
            }
            None 
        };

        // Finds root, None if Root was unable to be found 
        let newtons = |y: &Polynomial<T>, dy: &Polynomial<T>, bounds: &(T, T)| -> Option<T> {
            let mut x = (bounds.0 + bounds.1) / num_traits::NumCast::from(2).unwrap();
            for _ in 1..max_iter {
                let fx  = y.evaluate(&x);
                // Early return for exact root at x0 
                if fx.abs() < tol {
                    return Some(x)
                }
                
                let dfx = dy.evaluate(&x);

                let x_n = x - (fx / dfx);
                if (x_n - x).abs() < tol {
                    return Some(x_n)
                }

                x = x_n;
            }
            None
        };

        let mut roots: Vec<T> = Vec::new();
        let mut y: Polynomial<T> = self.clone();

        // Collect roots iteratively 
        for _ in 1..max_iter {
            // Get derivative 
            let dy: Polynomial<T> = Polynomial {
                coeff: derivative(&y.coeff) 
            };
            
            let bounds = bracket(&y).unwrap_or_else(|| panic!("No Bracket Exists")); 
            let root = newtons(&y, &dy, &bounds).unwrap_or_else(|| panic!("Complex Roots.")); 

            println!("Root: {}", root);
            roots.push(root);

            // Reduce degree of polynomial and trim degree 
            y.coeff = poly_divide(&y.coeff, root);
            y.trim();
            println!("{}", y);

            // Polynomial sufficiently reduced
            if y.coeff.len() == 1 {
                break;
            }
        }
        roots
    }
}
fn main() {
    let p: Polynomial<f32> = [-120.0, 274.0, -225.0, 85.0, -15.0, 1.0].into();
    println!("{}", p);
    println!("Roots: {:?}", p.get_roots())
}
