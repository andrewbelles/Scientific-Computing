use std::fmt;
use std::ops;
use std::iter;

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

// Idiomatic indexing for matrix class 
impl<T> ops::Index<(usize, usize)> for Matrix<T> {
    type Output = T;

    fn index(&self, index: (usize, usize)) -> &T {
        let (row, col) = index; 
        let cols = self.dims[1];
        // Bounds check 
        assert!(row < self.dims[0] && col < cols, 
            "Index out of bounds: ({},{}) for {:?}", row, col, self.dims);
        &self.data[row * cols + col]
    }
}
impl<T> ops::IndexMut<(usize, usize)> for Matrix<T> {
    // For mutable borrow 
    fn index_mut(&mut self, index: (usize, usize)) -> &mut T {
        let (row, col) = index; 
        let cols = self.dims[1];
        // Bound checks 
        assert!(row < self.dims[0] && col < cols, 
            "Index out of bounds: ({},{}) for {:?}", row, col, self.dims);
        &mut self.data[row * cols + col]
    }
}

impl<T> ops::Add for Matrix<T>
where T:
    Clone 
    + ops::Add<Output = T>
{
    type Output = Matrix<T>;

    fn add(self, other: Matrix<T>) -> Matrix<T> {
        assert!(self.dims == other.dims, 
            "Dimension mismatch: {:?}, {:?}", self.dims, other.dims);

        // Incorparate each matching scalar b and sum with self
        let result = self.data.into_iter()
            .zip(other.data)
            .map(|(a, b)| a + b)
            .collect();

        // Return result 
        Matrix {
            dims: self.dims,
            data: result 
        }
    }
}

// Custom traits 
impl<T> Matrix<T> 
where T: 
    Copy
    + num_traits::Float
    + PartialOrd 
    + ops::Add<Output = T>
    + ops::Sub<Output = T> 
    + ops::Mul<Output = T>
    + ops::Div<Output = T> 
{
    fn t(&self) -> Matrix<T> {
        let (r, c) = (self.dims[0], self.dims[1]);
        let mut data = Vec::with_capacity(r * c);

        // for each col j, for each row i, push M[i,j]
        for j in 0..c {
            for i in 0..r {
                let index = i * c + j;
                data.push(self.data[index]);
            }
        }

        // new dims are [cols, rows]
        Matrix { dims: [c, r], data }
    }
}



fn main() {
    let a: Matrix<f32> = [
        [1.0, 2.0, 3.0], 
        [4.0, 5.0, 6.0], 
        [7.0, 8.0, 9.0]
    ].into();
    
    println!("{:?}", a);
    println!("{:?}", a.t());

}
