use std::fmt;
use std::iter;
use std::ops;

#[derive(Debug, Clone)]
struct Matrix<T> {
    dims: [usize; 2],
    data: Vec<T> 
}


// Default 0x0 empty matrix 
impl<T> Default for Matrix<T> {
    fn default() -> Self {
        Matrix {
            dims: [0, 0],
            data: Vec::new()
        }
    }
}

// Interface for simplified call to generate mxn matrix 
impl<T: Clone, const R: usize, const C: usize> From<[[T; C]; R]> for Matrix<T> {
    fn from(array: [[T; C]; R]) -> Self {
        let data = array.iter()
            .flatten()
            .cloned()
            .collect();
        Matrix { dims: [R, C], data }
    }
}

impl<T: Clone> Matrix<T> {
    fn t(&self) -> Matrix<T> {
        let (r, c) = (self.dims[0], self.dims[1]);
        let mut data = Vec::with_capacity(r * c);

        // for each col j, for each row i, push M[i,j]
        for j in 0..c {
            for i in 0..r {
                let index = i * c + j;
                data.push(self.data[index].clone());
            }
        }

        // new dims are [cols, rows]
        Matrix { dims: [c, r], data }
    }
}

// Polynomial class over real numbers 
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

// 
impl<T: Clone> Polynomial<T>
where 
    T: Copy 
    + iter::Sum
    + num_traits::Float
    + num_traits::NumCast
{
    // Evaluate the polynomial at some x value 
    fn evaluate(&self, x: &T) -> T {
        // Multiplies each coefficient by the desired x coordinate to respective power i 
        self.coeff.iter().enumerate()
            .map(|(i, c)| {
                *c * x.powi(i as i32)
            })
            .sum()
    }
    // Return vector of roots to polynomial 
    fn get_roots(&self) -> Vec<T> {
        let tol: T = num_traits::NumCast().from(1e-3).unwrap(); 
        const MAX_ITER: u32 = 100;

        let mut roots: Vec<T> = Vec::new();
        let mut y: Polynomial<T> = self.clone();

        // Helper function to get derivative of function
        fn derivative<T>(poly: &[T]) -> Vec<T> 
        where T: Clone + ops::Mul<Output = T> + num_traits::NumCast {
            // Applies power rule to each coefficient and down shifts such that matrix is 1 small
            poly.iter().enumerate()
                .skip(1)        // Skip constant value = 0 
                .map(|(i, c)| {
                    let ti: T = num_traits::NumCast::from(i).unwrap();
                    c.clone() * ti
                })
            .collect()
        }

        // Helper function to perform synthetic division with a real valued root 
        fn poly_divide<T>(poly: &[T], root: T) -> Vec<T> 
        where T: Copy + ops::Mul<Output = T> + ops::Add<Output = T> {
            let degree = poly.len() -  1;
            let mut res: Vec<T> = Vec::with_capacity(degree);
            res.push(poly[0]);

            // Divides out polynomial
            for i in 1..=degree {
                res.push(poly[i] + root * res[i - 1]); 
            }
            res // Ignore remainder -> Assume 0 since dividing root off  
        }

        // Returns a tuple of good lower and upper bounds 
        fn bound<T>(poly: &Polynomial<T>, dx: T) -> Option<(T, T)>
        where T: 
            Copy 
            + num_traits::NumCast
            + num_traits::Float
            + iter::Sum
        {
            let x0: T = T::zero();
            let f0: T = poly.evaluate(&x0);

            // Naive bracketing loop for set number of iterations 
            for i in 0..MAX_ITER {
                let k: T = num_traits::NumCast::from(i).unwrap();
                let a: T = x0 - k * dx;   
                let b: T = x0 + k * dx;

                let fa: T = poly.evaluate(&a);
                let fb: T = poly.evaluate(&b);

                // On off chance we found a root return it as a tuple 
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
                }
            }

            None 
        }

        // Finds root, None if Root was unable to be found 
        fn newtons<T>(y: &Polynomial<T>, dy: &Polynomial<T>, bounds: (T, T)) -> Option<T> 
        where T:
            Copy
            + num_traits::NumCast
            + num_traits::Float
            + iter::Sum
        {
            let mut x = (bounds.0 + bounds.1) / num_traits::NumCast::from(2.0).unwrap();
            let fx  = y.evaluate(&x);
            if fx.abs() < tol {
                return Some(x)
            }
            
            let dfx = dy.evaluate(&x);

            let x_n = x - (fx / dfx);
            if (x_n - x).abs() < tol {
                return Some(x_n)
            }

            x = x_n;

            None
        }


        for i in 1..MAX_ITER {
            // Get derivative 
            let dy: Polynomial<T> = Polynomial {
                coeff: derivative(&y.coeff) 
            };



        }

        roots
    }
}

fn main() {
    let a: Matrix<f32> = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]].into();
    println!("{:?}", a);
    println!("{:?}", a.t());

    let p: Polynomial<f32> = [1.2, -4.5, 1.1, 3.0].into();
    println!("{}", p);
    println!("Value: {}", p.evaluate(&1.0));
}
    
