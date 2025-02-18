// Simple matrix class that encloses data in 2D flattened vector (to 1D)
#[derive(Clone)]
pub struct Matrix {
    row: usize,
    col: usize,
    data: Vec<f64> 
}

// C style (Row Major) Instantiate Trait Implementation 
pub trait Instantiate { fn new(value: f64, row: usize, col: usize) -> Self; }
impl Instantiate for Matrix {
    fn new(value: f64, row: usize, col: usize) -> Self {
        // Instantiates data to vector sized row * col and returns a new Matrix 
        let data = vec![value; row * col];
        Matrix { row, col, data }       // Return new matrix
    }
}

// Print value of matrix to stdout
pub trait Print { fn print(&self); }
impl Print for Matrix {
    // Iterates row major and prints values per row
    fn print(&self) {
        for i in 0..self.row {
            print!("[ ");
            for j in 0..self.col {
                print!("{} ", self.data[i * self.col + j]);
            }
            println!("]");
        }
    }
}

// Place vector into matrix: Convert 1D vector to 2D matrix 
pub trait Fill { fn fill(&mut self, vector: &[f64]) -> Option<bool>; }
impl Fill for Matrix {
    fn fill(&mut self, vector: &[f64]) -> Option<bool> {
        match vector.len() == self.row * self.col {
            false => return None,   // Only invalid if sizes aren't valid 
            true  => {
                // Places associated value from vector into matrix data section
                for i in 0..self.row {
                    for j in 0..self.col {
                        let k = i * self.col + j;
                        self.data[k] = vector[k];
                    }
                }
                return Some(true)
            }
        }
    }
}

// Implementation of naive matrix multiplication ( O(n^3) )
pub trait Matmul { fn mul(&mut self, matrix: &Matrix) -> Option<Matrix>; }
impl Matmul for Matrix {
    fn mul(&mut self, matrix: &Matrix) -> Option<Matrix> {
        match self.col == matrix.row {
            false => return None,   // Only can return none if sizes are invalid 
            true  => {
                // Sizes match proceed with naive matmul
                let mut result = Matrix::new(0.0, self.row, matrix.col);

                // Iterates over outer sizes and shared size
                for i in 0..self.row {
                    for j in 0..matrix.col {
                        // Shared size
                        for k in 0..self.col {
                            // Sum dot product
                            result.data[i * result.col + j] += self.data[i * self.col + k] * matrix.data[k * matrix.col + j];
                        }
                    }
                }
                return Some(result) 
            }
        }

    }
}

// Transpose implementation for both square and non-square matrices 
pub trait Transpose { fn transpose(&mut self); }
impl Transpose for Matrix {
    fn transpose(&mut self) {

        match self.row == self.col {
            true  => {
                // Handle swap for in-place square matrix
                for i in 0..self.row {
                    for j in i + 1..self.row {
                        let temp = self.data[i * self.col + j];
                        self.data[i * self.col + j] = self.data[j * self.col + i];
                        self.data[j * self.col + i] = temp;
                    }
                }
                // No reason to swap indices as they match
            },
            false => {
                let mut copy = Vec::with_capacity(self.row * self.col);
                
                for i in 0..self.col {
                    for j in 0..self.row {
                        copy.push(self.data[j * self.col + i]);
                    }
                }

                // Swap values for non-square matrix 
                self.row ^= self.col;
                self.col ^= self.row;
                self.row ^= self.col;
            }
        }
    }
}


fn main() {
    // Let matrices be able to be multiplied against each other
    let mut matrix = Matrix::new(0.0, 4, 3);
    let mut operator = Matrix::new(0.0, 3, 8);

    // Output vector dimensions 
    println!("Matrix Dim: {}x{}", matrix.row, matrix.col);
    println!("Operator Dim: {}x{}", operator.row, operator.col);

    // Initialize random vectors as test
    let mut init_vec_1: Vec<f64> = vec![0.0; matrix.row * matrix.col];
    let mut init_vec_2: Vec<f64> = vec![0.0; operator.row * operator.col]; 
    for i in 0..init_vec_1.len() {
        init_vec_1[i] = rand::random_range(0.0..=10.0);
    }
    for i in 0..init_vec_2.len() {
        init_vec_2[i] = rand::random_range(0.0..=10.0);
    }

    // Fill the matricees with initalizing vectors 
    matrix.fill(&init_vec_1);
    operator.fill(&init_vec_2);

    // Take the option and handle 
    let result_matrix: Matrix = match matrix.mul(&operator) {
        None => {
            panic!("Invalid Operand Sizes!");
        },
        Some(result) => result,
    };
    result_matrix.print();

    // Let both matrices be equal in value
    matrix = Matrix::new(0.0, 1, 4);
    operator = Matrix::new(0.0, 1, 4);

    // Show that current dimensions are not valid for matmul
    println!("Matrix Dim: {}x{}", matrix.row, matrix.col);
    println!("Operator Dim: {}x{}", operator.row, operator.col);

    // Initialize new vectors
    let mut init_vec_1 = vec![0.0; matrix.row * matrix.col];
    let mut init_vec_2 = vec![0.0; operator.row * operator.col]; 
    for i in 0..init_vec_1.len() {
        init_vec_1[i] = rand::random_range(0.0..=10.0);
    }
    for i in 0..init_vec_2.len() {
        init_vec_2[i] = rand::random_range(0.0..=10.0);
    }

    // Fill vectors 
    matrix.fill(&init_vec_1);
    operator.fill(&init_vec_2);
    // Take transpose 
    operator.transpose();

    // Capture results and print if valid 
    let result_matrix: Matrix = match matrix.mul(&operator) {
        None => {
            panic!("Invalid Operand Sizes")
        },
        Some(result) => result,
    };
    result_matrix.print();
}
