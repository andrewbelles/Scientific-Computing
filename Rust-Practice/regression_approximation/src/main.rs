use std::fs::File;
use std::io::{self, BufReader, BufRead, Write};
use std::path::PathBuf;
use std::env;
use std::f64::consts::PI;
use rand::distr::{Uniform, Distribution};

mod rewrite;


// Simple matrix class that encloses data in 2D flattened vector (to 1D)
#[derive(Clone)]
pub struct Matrix {
    row: usize,
    col: usize,
    data: Vec<f64> 
}

// C style (Row Major) Instantiate Trait Implementation 
pub trait InstantiateMatrix { fn new(value: f64, row: usize, col: usize) -> Self; }
impl InstantiateMatrix for Matrix {
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

// Implements add for matrices that are correctly sized 
pub trait MatAdd { fn add(&mut self, matrix: &Matrix) -> &mut Self; }
impl MatAdd for Matrix {
    fn add(&mut self, matrix: &Matrix) -> &mut Self {
        // Direct add
        if self.row == matrix.col || self.col == matrix.row {
            if self.col == matrix.row && self.row == matrix.row {
                for i in 0..self.row {
                    for j in 0..self.col {
                        self.data[i * self.col + j] += matrix.data[i * self.col + j];
                    }
                }
            // Matrix is single column array. Add to each column of row  
            } else if matrix.col == 1 {
                for i in 0..self.row {
                    for j in 0..self.col {
                        self.data[i * self.col + j] += matrix.data[j];
                    }
                }
            // Or single row array likewise 
            } else if matrix.row == 1 {
                for i in 0..self.row {
                    for j in 0..self.col {
                        self.data[i * self.col + j] += matrix.data[i];
                    }
                }
            }
        } else if self.row == matrix.row {
            if matrix.col == 1 {
                for i in 0..self.row {
                    for j in 0..self.col {
                        self.data[i * self.col + j] += matrix.data[i];
                    }
                }
            } 
        } else {
            panic!("Dimension Mismatch: {}x{} and {}x{}", self.row, self.col, matrix.row, matrix.col);
        }
        self
    }
}

// Inverse implementation of matrix subtraction that follows the same pattern
// Implements add for matrices that are correctly sized 
pub trait MatSub { fn sub(&mut self, matrix: &Matrix) -> &mut Self; }
impl MatSub for Matrix {
    fn sub(&mut self, matrix: &Matrix) -> &mut Self {
        // Direct add
        if self.row == matrix.col || self.col == matrix.row {
            if self.col == matrix.row && self.row == matrix.row {
                for i in 0..self.row {
                    for j in 0..self.col {
                        self.data[i * self.col + j] -= matrix.data[i * self.col + j];
                    }
                }
            // Matrix is single column array. Add to each column of row  
            } else if matrix.col == 1 {
                for i in 0..self.row {
                    for j in 0..self.col {
                        self.data[i * self.col + j] -= matrix.data[j];
                    }
                }
            // Or single row array likewise 
            } else if matrix.row == 1 {
                for i in 0..self.row {
                    for j in 0..self.col {
                        self.data[i * self.col + j] -= matrix.data[i];
                    }
                }
            }
        } else if self.row == matrix.row {
            if matrix.col == 1 {
                for i in 0..self.row {
                    for j in 0..self.col {
                        self.data[i * self.col + j] -= matrix.data[i];
                    }
                }
            } 
        } else {
            panic!("Dimension Mismatch: {}x{} and {}x{}", self.row, self.col, matrix.row, matrix.col);
        }
        self
    }
}

// Implementation of naive matrix multiplication ( O(n^3) )
pub trait Matmul { fn mul(&mut self, matrix: &Matrix) -> Option<Matrix>; }
impl Matmul for Matrix {
    fn mul(&mut self, matrix: &Matrix) -> Option<Matrix> {
        match self.col == matrix.row {
            false => {
                panic!("Matmul Size Mismatch: {}x{} x {}x{}", self.row, self.col, matrix.row, matrix.col);
            },   // Only can return none if sizes are invalid 
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

pub trait Matscale { fn scale(&mut self, scale_factor: f64) -> Matrix; }
impl Matscale for Matrix {
    fn scale(&mut self, scale_factor: f64) -> Matrix {
        for i in 0..self.row {
            for j in 0..self.col {
                self.data[i * self.col + j] *= scale_factor;
            }
        }
        let result = self.clone();
        result
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
                // Move copied data back
                //
                self.data = copy;
                // Swap values for non-square matrix
                std::mem::swap(&mut self.row, &mut self.col);
            }
        }
    }
}

pub trait MLOperations {
    fn hadamard(&self, matrix: &Matrix) -> Matrix;
    fn mean(&self) -> f64;
    fn apply_derivative(&mut self, func: fn(f64) -> f64);
}
impl MLOperations for Matrix {
    // Multiply every data value in self with derivative matrix 
    fn hadamard(&self, matrix: &Matrix) -> Matrix {
        // Assertion logic for nxm matching size parameters 
        assert_eq!(self.row, matrix.row);
        assert_eq!(self.col, matrix.col);
        
        // Since sizes match (as asserted)
        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] *= matrix.data[i];
        }
        result
    }
    // Simply sums matrix data and returns average/mean
    fn mean(&self) -> f64 {
        self.data.iter().sum::<f64>() / ((self.row * self.col) as f64)
    }
    fn apply_derivative(&mut self, func: fn(f64) -> f64) {
        for i in 0..self.row {
            for j in 0..self.col {
                self.data[i * self.col + j] = func(self.data[i * self.col + j]);
            }
        }
    }
}

pub struct NeuronMatrix {
    weights: Matrix,
    biases:  Matrix, 
    w_count: usize,
    b_count: usize
}

// Access weights and biases mutable values 
impl NeuronMatrix {
    // I want weights and biases when accessed to be mutable _always_
    pub fn weights(&mut self) -> &mut Matrix {
        &mut self.weights
    }
    pub fn biases(&mut self) -> &mut Matrix {
        &mut self.biases 
    }
}

pub trait InstantiateNeuronMatrix { 
    fn new(neuron_ct: usize, prev_input_ct: usize) -> NeuronMatrix; 
    fn initialize(&mut self, sizes: &[usize; 2]); 
}
impl InstantiateNeuronMatrix for NeuronMatrix {
 
    fn new(neuron_ct: usize, prev_input_ct: usize) -> NeuronMatrix {
        NeuronMatrix {
            weights: Matrix::new(0.0, prev_input_ct, neuron_ct),
            biases : Matrix::new(0.0, 1, neuron_ct),
            w_count: prev_input_ct * neuron_ct, 
            b_count: neuron_ct
        }
    }
    // Sizes 0 is input count; Sizes 1 is output size 
    fn initialize(&mut self, sizes: &[usize; 2]) {
        // Calculate uniform distribution for input output size 

        //let uniform_range: f64 = (6.0 / ((sizes[0] + sizes[1]) as f64)).sqrt();
        //let uniform_range: f64 = (2.0 / (sizes[0] as f64)).sqrt(); 
        let uniform_range: f64 = (1.0 / (sizes[0] as f64)).sqrt();

        let uniform = Uniform::new_inclusive(-uniform_range, uniform_range).expect("failure");
        let mut rng = rand::rng();

        let weights = self.weights();
        for i in 0..weights.row {
            for j in 0..weights.col {
                let weight = uniform.sample(&mut rng); 
                weights.data[i * weights.col + j] = weight;
            }
        }   
        // Lifetime of weights borrow ends allowing us to borrow biases now 
        let biases  = self.biases();
        biases.fill(&vec![0.0; biases.col * biases.row]);
        biases.transpose();
    }
}

// Follows Structure of "Array"
// Stores information about ith network layer including type, 2d size, and encoded data in each
pub struct Network {
    layers: Vec<NeuronMatrix>,
    activations: Vec<Matrix>,
    sizes:  Vec<[usize; 2]>, 
    layer_count: usize 
}

// We aren't in charge of the input layer. It should be able to interface but isn't contained 
//
// Assume sizes holds for each layer the nxm size (n being in column 0 and m being in col 1)
pub trait InstantiateNetwork { fn new(input_ct: usize, output_ct: usize, neuron_counts: Vec<usize> ) -> Network; }
impl InstantiateNetwork for Network {
    fn new(input_ct: usize, output_ct: usize, neuron_counts: Vec<usize> ) -> Network {
        let count = neuron_counts.len() + 1; 
        let hidden_ct = neuron_counts.len(); 
         
        // Assume layers haven't been created yet
        let mut output_layer  = NeuronMatrix::new(output_ct, neuron_counts[hidden_ct - 1]);
        output_layer.initialize(&[neuron_counts[hidden_ct - 1], 1]);

        // Create hidden layers 
        let mut new_layers: Vec<NeuronMatrix> = Vec::new();
        let activations: Vec<Matrix> = Vec::new();
        let mut new_sizes: Vec<[usize; 2]> = Vec::new();        // [previous size, current_size]

        // Initialize first hidden layer as seperate? 
        let mut layer = NeuronMatrix::new(neuron_counts[0], input_ct);
        layer.initialize(&[layer.w_count, layer.b_count]);
        
        // Create first hidden layer outside loop
        new_layers.push(layer);
        new_sizes.push([input_ct, neuron_counts[0]]);

        // Start at 2nd hidden layer 
        for i in 1..hidden_ct {
            let prev_input = neuron_counts[i - 1];
            println!("Prev Count {}", prev_input);
            
            // Initialize weights and biasees for ith layer
            let mut layer = NeuronMatrix::new(neuron_counts[i], prev_input);
            layer.initialize(&[layer.w_count, layer.b_count]);

            // Push layer into network
            new_layers.push(layer);
            new_sizes.push([neuron_counts[i - 1], neuron_counts[i]]);
        }
        
        // Push the output layer into the network
        new_layers.push(output_layer);
        // Get last position and add final size (which reflects the matrix dimensions)
        new_sizes.push([*neuron_counts.last().unwrap(), output_ct]);

        return Network { layers: new_layers, activations, sizes: new_sizes, layer_count: count }
    }
}

// relu(z) => max(0, z)
pub fn rectified_linear_unit(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    } else {
        return x;
    }
}

pub fn relu_derivative(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.0;
    } else {
        return 1.0;
    }
}

pub fn leaky_relu(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.01 * x;
    } else {
        return x;
    }
}

pub fn leaky_relu_derivative(x: f64) -> f64 {
    if x <= 0.0 {
        return 0.01;
    } else {
        return 1.0;
    }
}

pub trait Activate { fn activate(&mut self, func: fn(f64) -> f64); }
impl Activate for Matrix {
    fn activate(&mut self, func: fn(f64) -> f64) {
        for i in 0..self.row {
            for j in 0..self.col {
                    self.data[i * self.col + j] = func(self.data[i * self.col + j]);
            }
        } 
    }
} 

// Implements a forward propagation through the network using flexible activation function. 
pub trait Forward { fn forward(&mut self, input: &mut Matrix, func: fn(f64) -> f64) -> Matrix; }
impl Forward for Network {
    fn forward(&mut self, input: &mut Matrix, func: fn(f64) -> f64) -> Matrix {
        self.activations.clear();

        let mut a = input.clone();
        self.activations.push(a.clone());

        // The first activation is the starting point of this loop 
        for (i,layer) in self.layers.iter_mut().enumerate() {
            let mut z = a.mul(layer.weights()).unwrap();
            z.add(&layer.biases());
            
            if i != self.layer_count-1 {
                z.activate(func);
            }
            a = z;
            self.activations.push(a.clone());
        }
        a
    }
}

pub trait Backward { fn backward(&mut self, expected: &Matrix , learning_rate: f64, func: fn(f64) -> f64); }
impl Backward for Network {
    fn backward(&mut self, expected: &Matrix, learning_rate: f64, func: fn(f64) -> f64) {
        // Copy intermediate value of delta and collect gradient of cost function a_L - y  
        let mut delta = self.activations.last().unwrap().clone();
        delta.sub(&expected);

        for i in 0..self.layer_count {
            // println!("Current Iteration: {i}"); 
            let current_layer_index = self.layer_count - i - 1;
            let activation_index    = current_layer_index + 1;

            // Find f'(a_i)
            let mut derivative = self.activations[activation_index].clone();
            derivative.apply_derivative(func);

            if i != 0 {
                let mut weight_c = self.layers[current_layer_index + 1].weights().clone();
                weight_c.transpose();
                // Update delta from (W_i+1)^T x delta_i+1
                delta = delta.mul(&weight_c).unwrap();
            }

            // println!("Hadamard Product");
            // find true next delta through hadamard 
            delta = delta.hadamard(&derivative);

            // Cost function: Input for layer (previous output or i - 2)
            let mut activation_c = self.activations[current_layer_index].clone();
            activation_c.transpose();
            // println!("Compute Weight Gradient");
            let mut weight_gradient = activation_c.mul(&delta).unwrap();
            // println!("Compute Bias Gradient");
            let mut bias_gradient   = Matrix::new(delta.mean(), delta.row, delta.col);

            // println!("Compute New Weights and Biases for Layer");
            self.layers[current_layer_index].weights().sub(&weight_gradient.scale(learning_rate));
            self.layers[current_layer_index].biases().sub(&bias_gradient.scale(learning_rate));
        }
    }
}

// Generates random sine wave under amplitude 
pub fn generate_sparse_data(size: &usize, ampl: &f64) -> Vec<[f64; 2]> {
    let mut data: Vec<[f64; 2]> = Vec::with_capacity(*size as usize);
    
    let step: f64   = (20.0*PI) / (*size as f64);
    let mut dx: f64 = 0.0;

    for _ in 0..=*size {
        let y = ampl * (dx.sin());
        data.push([dx, y]);
        dx += step;
    }

    let mut sparse: Vec<[f64; 2]> = Vec::new();

    // Create Sparse Data matrix and return it 
    for i in 0..data.len() {
        let b = rand::random_bool(0.005);
        if b == true {
            sparse.push(data[i]);
        }
    }
    return sparse
}

pub fn parse_val(arg: &str, prefix: &str) -> Option<usize> {
    let slice = &arg[prefix.len()..];
    slice.trim().parse::<usize>().map_err(|e| {
        eprintln!("Error {e}");
    }).ok()
}

pub fn string_to_int(arg: &str) -> Option<usize> {
    if arg.starts_with("size=") {
        parse_val(arg, "size=")
    } else if arg.starts_with("ampl=") {
        parse_val(arg, "ampl=")
    } else {
        None 
    }
}

// Parses [a, b, c, etc] into the counts
pub fn parse_counts(arg: &str) -> Option<Vec<usize>> {
    if !arg.starts_with('[') || !arg.ends_with(']') {
        return None
    }

    let arg_tr = arg.trim_start_matches('[').trim_end_matches(']');
    if arg_tr.is_empty() {
        return Some(Vec::new())
    }

    let counts: Result<Vec<usize>, _> = arg_tr
        .split(',')
        .map(|count_str| {
            count_str.trim().parse::<usize>().map_err(|_| {
                "Parse Failure!".to_string()
        })
    }).collect();

    match counts {
        Ok(vec) => Some(vec),
        Err(_) => None
    }
}

pub fn save_epoch_output(epoch: usize, data: Vec<f64>) -> io::Result<()> {
    
    let filename = format!("epoch_{:04}.txt", epoch);
    let mut path = PathBuf::from("./output_files");
    path.push(filename);

    let mut file = File::create(&path)?;
    for i in 0..data.len() {
        writeln!(file, "{}", data[i])?;
    }

    Ok(())
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    match args.len() {
        5 if args[1] == "gen" => {
            let size_arg = &args[2]; // &str 
            let ampl_arg = &args[3];
            let output   = &args[4]; // "

            if let (Some(size), Some(ampl)) = (string_to_int(size_arg), string_to_int(ampl_arg)) {
                println!("Generating Test Data...");

                let data = generate_sparse_data(&size, &(ampl as f64)); 
                let mut file = File::create(output)?;
                let values = &data[..];
                for [x,y] in values.iter() {
                    write!(&mut file, "{}", x)?;
                    writeln!(&mut file, ",{}", y)?;
                }
            } else {
                panic!("Invalid Arguments. Usage: [cargo run] [gen] [size=] [ampl=] [sparse_data.txt]");
            }
            return Ok(())
        },
        _ => {
            if args.len() != 4 {
                eprintln!("Invalid Arguments Usage: [cargo run] [sparse/training_data.txt] [counts] [t/s]");
                eprintln!("Alternative: [cargo run] [gen] [size=] [ampl=] [sparse_data.txt]");
                return Ok(());
            }
        }
    }

    let input_path = &args[1];
    let count_arg = &args[2];
    let counts = parse_counts(count_arg).unwrap();
    // let mode       = &args[2];

    // Read data into array
    let file    = File::open(input_path)?;
    let reader  = BufReader::new(file);
    let mut data: Vec<[f64; 2]> = Vec::new();

    for line in reader.lines() {
        let line = line?;
        let line = line.trim();

        let coordinate: Vec<&str> = line.split(',').collect();
        if coordinate.len() == 2 {
            // Pull x and y strings 
            let x_ = coordinate[0].trim();
            let y_ = coordinate[1].trim();

            match (x_.parse::<f64>(), y_.parse::<f64>()) {
                (Ok(x), Ok(y)) => {
                    data.push([x, y]);
                }
                _ => {
                    eprintln!("Parse Error");
                    return Ok(());
                }
            }
        }
    }

    // pass data to the matching call -t or -o
    // -o also needs to pull the weights.txt file
    let use_arg = &args[3];
    if use_arg == "-t" {

        let size = data.len();

        let mut input_data   = Matrix::new(0.0, size, 1);
        let mut expected     = Matrix::new(0.0, 1, size);

        let mut input_vec  = vec![0.0; size];
        let mut expect_vec = vec![0.0; size]; 

        for i in 0..size {
            input_vec[i]  = data[i][0];
            expect_vec[i] = data[i][1];
        }

        expected.fill(&expect_vec);
        input_data.fill(&input_vec);

        // Make example network.
        
        let mut network = Network::new(1, 1, counts);
        println!("Size {:?}", network.sizes);

        let learning_rate = 1e-3;
        let epochs = 1000;

        for epoch in 0..epochs {
            let mut output = network.forward(&mut input_data, |x| x.tanh() );

            let loss_matrix = output.sub(&expected);
            let loss = loss_matrix.data.iter().map(|x| x.powi(2)).sum::<f64>() / (size as f64);
        
            network.backward(&expected, learning_rate, |x| 1.0 - x.tanh().powi(2));

            if epoch % 1 == 0 {
                println!("Epoch {} Loss {}", epoch, loss);
            }
            
            // Save data into output_files
            save_epoch_output(epoch, output.data).unwrap();
        }

        let mut file = File::create("weights.txt")?;
        for layer in network.layers.iter_mut() {
            
            let weight_data = &layer.weights().data;
            for weight in weight_data.iter() {
                writeln!(&mut file, "{}", weight)?;
            }

            let bias_data = &layer.biases().data;
            for bias in bias_data.iter() {
                writeln!(&mut file, "{}", bias)?;
            }
        }
    } else if use_arg == "-o" {
        
        let file = File::open("weights.txt")?;
        let reader  = BufReader::new(file);
        
        let mut load_h_weights: Vec<f64> = Vec::with_capacity(8);
        let mut load_h_biases : Vec<f64> = Vec::with_capacity(8);
        let mut load_o_weights: Vec<f64> = Vec::with_capacity(8);
        let mut load_o_biases : Vec<f64> = Vec::with_capacity(1);

        for (i, line) in reader.lines().enumerate() {
            let line = line?;
            let line = line.trim();
            
            let value = match line.parse::<f64>() {
                Ok(value) => value,
                Err(e) => panic!("Parse Error {e}"),
            };

            // Load hidden weights 
            if i < 8 {
               load_h_weights.push(value); 
            // Load hidden biases
            } else if i < 16 {
                load_h_biases.push(value);
            // Load output weights
            } else if i < 24 {
                load_o_weights.push(value);
            // Load output bias
            } else {
                load_o_biases.push(value);
            }
        }

        let mut network = Network::new(1, 1, [8].to_vec());
        network.layers[0].weights().fill(&load_h_weights); 
        network.layers[0].biases().fill(&load_h_biases); 
        network.layers[1].weights().fill(&load_o_weights); 
        network.layers[1].biases().fill(&load_o_biases); 

        // Create input vector to pass through network 
        let step: f64   = (20.0*PI) / (10000 as f64);
        let mut dx: f64 = 0.0;

        let mut inputs: Vec<f64> = Vec::new();
        for _ in 0..=10000 {
            inputs.push(dx);
            dx += step;
        }
        let mut input_matrix = Matrix::new(0.0, 10000, 1);
        input_matrix.fill(&inputs);
        let output = network.forward(&mut input_matrix, leaky_relu);

        let mut data: Vec<[f64; 2]> = Vec::new();
        for i in 0..10000 {
            data.push([inputs[i], output.data[i]]);
        }

        let mut file = File::create("./prediction/output.txt")?;
        let values = &data[..];
        for [x,y] in values.iter() {
            write!(&mut file, "{}", x)?;
            writeln!(&mut file, ",{}", y)?;
        }
        
    } else {
        eprintln!("Invalid Arguments Usage: [cargo] [sparse/training_data.txt] [t/s]");
        eprintln!("Alternative: [cargo run] [gen] [size=] [ampl=] [sparse_data.txt]");
    } 

    return Ok(());
}
