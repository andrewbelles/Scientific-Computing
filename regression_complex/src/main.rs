// Design choices that differ in this one. 
// I want to all methods acting between two matrices to return a new Matrix
//
// The network should encode the selected function and its derivative within it. 
//   It will be up to initialize the function. 
mod activations;

use std::fs::File;
use std::io::{self, BufReader, BufRead, Write};
use std::path::PathBuf;
use std::env;
use rand::distr::{Uniform, Distribution};
use crate::activations::{*, ActivationType::*};

// MATRIX OPERATIONS AND LAYER/NETWORK CREATION/USAGE

#[derive(Clone)]
pub struct Matrix {
    row: usize,
    col: usize,
    data: Vec<f64>
}

// Simple instantiate that initializes a Matrix of row x col to a certain value 
pub trait InstantiateMatrix { fn new(value: f64, row: usize, col: usize) -> Self; }
impl InstantiateMatrix for Matrix {
    fn new(value: f64, row: usize, col: usize) -> Self {
        let data = vec![value; row * col];
        Matrix { row, col, data }
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

// All intrinsic Matrix Methods
pub trait MatrixOperations{
    fn scale(&self, scalar: f64) -> Matrix;
    fn transpose(&self) -> Matrix;
    fn add(&self, matrix: &Matrix) -> Option<Matrix>; 
    fn sub(&self, matrix: &Matrix) -> Option<Matrix>;
    fn mul(&self, matrix: &Matrix) -> Option<Matrix>;
    fn mean(&self) -> f64;
    fn stdev(&self, mean: &mut f64) -> f64;
}
impl MatrixOperations for Matrix {
    // Scale matrix by some scalar value 
    fn scale(&self, scalar: f64) -> Matrix {
        let mut result = Matrix::new(0.0, self.row, self.col);
        for i in 0..self.row {
            for j in 0..self.col {
                result.data[i * self.col + j] = self.data[i * self.col + j] * scalar;
            }
        }
        result
    }
    // Tranpose Both Square and Non-square matrices 
    fn transpose(&self) -> Matrix {
        let mut copy = Matrix::new(0.0, self.col, self.row);
        for i in 0..self.row {
            for j in 0..self.col {
                copy.data[j * self.row + i] = self.data[i * self.col + j];
            }
        }
        copy
    }
    // Add two matrices after ensure their sizes are acceptable
    fn add(&self, matrix: &Matrix) -> Option<Matrix> {
        if self.row == matrix.row && self.col == matrix.col {
            let mut result = Matrix::new(0.0, self.row, self.col);
            for i in 0..self.row {
                for j in 0..self.col {
                    result.data[i * self.col + j] = self.data[i * self.col + j] + matrix.data[i * self.col + j];
                }
            }
            return Some(result);
        } else if matrix.row == 1 && matrix.col == 1 {
            let mut result = Matrix::new(0.0, self.row, self.col);
            for i in 0..self.row {
                for j in 0..self.col {
                    result.data[i * self.col + j] = self.data[i * self.col + j] + matrix.data[0];
                }
            }
            return Some(result);
        } else if matrix.row == 1 && matrix.col == self.col {
            let mut result = Matrix::new(0.0, self.row, self.col);
            for i in 0..self.row {
                for j in 0..self.col {
                    result.data[i * self.col + j] = self.data[i * self.col + j] + matrix.data[j];
                }
            }
            return Some(result);
        } else if matrix.col == 1 && matrix.row == self.row {
            let mut result = Matrix::new(0.0, self.row, self.col);
            for i in 0..self.row {
                for j in 0..self.col {
                    result.data[i * self.col + j] = self.data[i * self.col + j] + matrix.data[i];
                }
            }
            return Some(result);
        }

        // Dimension mismatch
        None
    }
    // Identical implementation just - from +
    fn sub(&self, matrix: &Matrix) -> Option<Matrix> {
        if self.row == matrix.row && self.col == matrix.col {
            let mut result = Matrix::new(0.0, self.row, self.col);
            for i in 0..self.row {
                for j in 0..self.col {
                    result.data[i * self.col + j] = self.data[i * self.col + j] - matrix.data[i * self.col + j];
                }
            }
            return Some(result);
        } else if matrix.row == 1 && matrix.col == 1 {
            let mut result = Matrix::new(0.0, self.row, self.col);
            for i in 0..self.row {
                for j in 0..self.col {
                    result.data[i * self.col + j] = self.data[i * self.col + j] - matrix.data[0];
                }
            }
            return Some(result);
        } else if matrix.row == 1 && matrix.col == self.col {
            let mut result = Matrix::new(0.0, self.row, self.col);
            for i in 0..self.row {
                for j in 0..self.col {
                    result.data[i * self.col + j] = self.data[i * self.col + j] - matrix.data[j];
                }
            }
            return Some(result);
        } else if matrix.col == 1 && matrix.row == self.row {
            let mut result = Matrix::new(0.0, self.row, self.col);
            for i in 0..self.row {
                for j in 0..self.col {
                    result.data[i * self.col + j] = self.data[i * self.col + j] - matrix.data[i];
                }
            }
            return Some(result);
        }

        // Dimension mismatch
        None
    }
    // Matrix multiplication 
    fn mul(&self, matrix: &Matrix) -> Option<Matrix> {
        // Check for valid inner indices
        match self.col == matrix.row {
            false => None,  // Only reason we might not be allowed to do matrix multiplication 
            true  => {
                let mut result = Matrix::new(0.0, self.row, matrix.col);

                for i in 0..self.row {
                    for j in 0..matrix.col {
                        for k in 0..self.col {
                            result.data[i * result.col + j] += self.data[i * self.col + k] * matrix.data[k * matrix.col + j];
                        }
                    }
                }
                Some(result)
            }
        }
    }
    // Take the sum of all data points in matrix and find average 
    fn mean(&self) -> f64 {
        let mut sum: f64 = 0.0;
        for i in 0..self.data.len() {
            sum += self.data[i];
        }
        sum /= (self.row * self.col) as f64;
        sum
    }
    fn stdev(&self, mean: &mut f64) -> f64 {
        if *mean == 0.0 {
            *mean = self.mean();
        }

        let mut residual_sum: f64 = 0.0;
        for i in 0..self.row {
            for j in 0..self.col {
                residual_sum += (self.data[i * self.col + j] - *mean).powi(2); 
            }
        }
        ((residual_sum) / (self.row * self.col - 1) as f64).sqrt()
    }
}

// Machine Learning specific operations for Matrix 
pub trait MachineLearningOperations {
    fn hadamard(&self, matrix: &Matrix) -> Option<Matrix>;
    fn activate(&self, func: fn(f64) -> f64) -> Matrix;
}
impl MachineLearningOperations for Matrix {
    // Find hadamard product by multiplying matrix a by the corresponding values it matrix b 
    fn hadamard(&self, matrix: &Matrix) -> Option<Matrix> {
        // a and b must be identical in size 
        if self.row != matrix.row || self.col != matrix.col {
            return None // Only reason it can't be successful is mismatched sizes
        }

        // Multiply against 
        let mut result = self.clone();
        for i in 0..self.data.len() {
            result.data[i] *= matrix.data[i];
        }
        Some(result) 
    }
    // Apply some derivative function to all elements of Matrix a 
    fn activate(&self, func: fn(f64) -> f64) -> Matrix {
        let mut derivative: Vec<f64> = vec![0.0; self.row * self.col];
        for i in 0..self.row {
            for j in 0..self.col {
                derivative[i * self.col + j] = func(self.data[i * self.col + j]);
            }
        }
        // Return new matrix 
        Matrix { row: self.row, col: self.col, data: derivative }
    }
}

#[derive(Clone)]
pub struct Layer {
    weights: Matrix,
    biases: Matrix,
    func: Activation
}

pub trait InstantiateLayer {
    fn new(neuron_count: usize, prev_input_count: usize, funcs: Activation) -> Layer;
}
impl InstantiateLayer for Layer {
    fn new(neuron_count: usize, prev_input_count: usize, funcs: Activation) -> Layer {
        // Initialize uniform range depending on function family used.
        let uniform_range: f64 = match funcs.activation_type {
            Tanh | Sigmoid => (6.0 / ((neuron_count + prev_input_count) as f64)).sqrt(),
            LeakyReLu | ReLu | ELU => (2.0 / (prev_input_count as f64)).sqrt(),
            _ => (6.0 / ((neuron_count + prev_input_count) as f64)).sqrt(),
        };

        let uniform = Uniform::new_inclusive(-uniform_range, uniform_range).unwrap();
        let mut rng = rand::rng();

        // Pull random value from distribution for each 
        let mut weights = Matrix::new(0.0, prev_input_count, neuron_count);
        for i in 0..weights.row {
            for j in 0..weights.col {
                let weight = uniform.sample(&mut rng);
                weights.data[i * weights.col + j] = weight;
            }
        }
        // Initlize biases to small nonzero value  
        let biases = Matrix::new(0.0, 1, neuron_count);

        Layer { weights, biases, func: funcs }
    }
}

// Network of Layers each containing Matrix of weights and array of biases
pub struct Network {
    layers: Vec<Layer>,
    inputs: Vec<Matrix>,
    activations: Vec<Matrix>,
    sizes: Vec<[usize; 2]>,
    layer_count: usize,
}

// Counts holds the number of neurons for each layer which is used to map each matrix as connecting 
//
// counts.first() and counts.last() hold the Input and Output Count respectively to simplify logic
pub trait InstantiateNetwork {
    fn new(counts: &Vec<usize>, funcs: Vec<Activation>) -> Network;
    //fn load(weight_path: String) -> Network;
}
impl InstantiateNetwork for Network {
    fn new(counts: &Vec<usize>, funcs: Vec<Activation>) -> Network {
        let layer_count                    = counts.len() - 1;
        let mut new_layers: Vec<Layer>     = Vec::new();
        let inputs: Vec<Matrix>            = Vec::with_capacity(layer_count);
        let activations: Vec<Matrix>       = Vec::with_capacity(layer_count + 1);   // Create with +1
        let mut new_sizes: Vec<[usize; 2]> = Vec::new();

        // Create each layer using input counts 
        for i in 0..layer_count {
            // Set Layer depending on first hidden or any subsequent
            let new_layer = Layer::new(counts[i + 1], counts[i], funcs[i].clone());
            new_sizes.push([counts[i], counts[i + 1]]);
            new_layers.push(new_layer);
        }
        // Return new network
        Network { layers: new_layers, inputs, activations, sizes: new_sizes, layer_count }
    }
    /*fn load(weight_path: String) -> Network {
        

    }*/
}

pub trait ForwardPropagation { fn forward(&mut self, input_matrix: &mut Matrix) -> Matrix; }
impl ForwardPropagation for Network {
    fn forward(&mut self, input_matrix: &mut Matrix) -> Matrix {
        // Clear any previous inputs 
        self.inputs.clear();
        self.activations.clear();

        let mut a = input_matrix.clone();
        self.activations.push(a.clone());

        // Comfortable iterating like so since we only need the current layer (unlike back prop)
        for (i, layer) in self.layers.iter_mut().enumerate() {
            // z = a_l-1*W_l + b_l 
            let z = a.mul(&layer.weights).unwrap()
                .add(&layer.biases).unwrap();

            self.inputs.push(z.clone());

            a = if i != self.layer_count - 1 {
                z.activate(layer.func.function)
            } else {
                z.clone()
            };
            
            self.activations.push(a.clone());
        }
        a
    }
}

pub fn clip_gradients(gradient: &Matrix, range: f64) -> Matrix {
    let mut clipped = Matrix::new(0.0, gradient.row, gradient.col);

    // Clamp values 
    for i in 0..(gradient.row * gradient.col) {
        clipped.data[i] = gradient.data[i].clamp(-range ,range);
    }
    clipped
}

pub trait BackwardPropagation { fn backward(&mut self, expected: &Matrix, learning_rate: f64); }
impl BackwardPropagation for Network {
    fn backward(&mut self, expected: &Matrix, learning_rate: f64) {
        // First delta is just output - expected values 
        let mut delta = self.activations.last().unwrap().sub(&expected).unwrap();

        // Iterate in reverse starting from output layer 
        for i in (0..self.layer_count).rev() {
            // Borrow layer
            let a_prev = &self.activations[i];

            // Gradient calculations

            // Weight Grad
            let weight_gradient = a_prev.transpose().mul(&delta).unwrap().scale(learning_rate);

            // Bias Grad
            let mut bias_gradient = Matrix::new(0.0, 1, delta.col);
            for j in 0..delta.col {
                let mut sum = 0.0;
                for k in 0..delta.row {
                    sum += delta.data[k * delta.col + j];
                }
                bias_gradient.data[j] = sum / delta.row as f64;
            }
            bias_gradient.scale(learning_rate);

            //weight_gradient = clip_gradients(&weight_gradient, 1.0);
            //bias_gradient   = clip_gradients(&bias_gradient, 1.0);

            // Update weights and biases 
            self.layers[i].weights = self.layers[i].weights.sub(&weight_gradient).unwrap();
            self.layers[i].biases  = self.layers[i].biases.sub(&bias_gradient).unwrap();

            if i > 0 {
                let z_prev = &self.inputs[i - 1];
                let derivative = z_prev.activate(self.layers[i - 1].func.derivative);

                delta = delta.mul(&self.layers[i].weights.transpose()).unwrap();
                delta = delta.hadamard(&derivative).unwrap();
            }
        }
    }
}

// DATA GENERATION AND INPUT HANDLING 

// Parses [1,32,16,8,1] to correct Network Size 
pub fn parse_counts(argument: &str) -> Option<Vec<usize>> {
    // Check for bracket input 
    if !argument.starts_with('[') || !argument.ends_with(']') {
        return None
    }

    let argument_trim = argument.trim_start_matches('[').trim_end_matches(']');
    // If no counts between brackets return None 
    if argument_trim.is_empty() {
        return None 
    }

    // Pattern to split comma separated size values
    let parsed_counts: Result<Vec<usize>, _> = argument_trim.split(',').map(|count_string| {
        count_string.trim().parse::<usize>().map_err(|_| {
            "Parse Failure".to_string()
        })
    }).collect();

    match parsed_counts {
        Ok(counts) => Some(counts),
        Err(_) => None 
    }
}

pub fn save_epoch_output(epoch: &usize, data: &Vec<f64>) -> io::Result<()> {
    let filename = format!("epoch_{:04}.txt", epoch);
    let mut path = PathBuf::from("./outputs/");
    path.push(filename);

    let mut file = File::create(&path)?;
    for i in 0..data.len() {
        writeln!(file, "{}", data[i])?;
    }

    Ok(())
}

pub fn save_weights(network: &Network, path: &String) -> io::Result<()> {
    let mut file = File::create(path)?;
    
    // The second value in sizes corresponds the value from [1,32,16,8,1]
    // Construct vector of these values 
    let mut original_input: Vec<usize> = Vec::new();
    for size in network.sizes.iter() {
        original_input.push(size[1]);
    }

    // Print original network size at top of file 
    write!(file, "[")?;
    write!(file, "{},", network.sizes[0][0])?;
    for i in 0..original_input.len() {
        if i != original_input.len() - 1 {
            write!(file, "{},", original_input[i])?;
        } else {
            write!(file, "{}", original_input[i])?;
        }
    }
    writeln!(file, "]")?;

    // Print weights and biases below the network size spec     
    for layer in network.layers.iter() {
        // print weights then biases for each layer  
        // no weights are printed for first layer so we can use sizes i and i - 1

        for i in 0..layer.weights.row {
            for j in 0..layer.weights.col {
                writeln!(file, "{}", layer.weights.data[i * layer.weights.col + j])?;
            }
        }

        for j in 0..layer.biases.col {
            writeln!(file, "{}", layer.biases.data[j])?;
        }
    }

    Ok(())
}

// Decay Learning Rate exponentially 
#[inline(always)]
pub fn exp_decay_learning_rate(initial_rate: f64, epoch: f64, decay_rate: f64) -> f64 {
    initial_rate * decay_rate.powi((epoch/ 10.0) as i32)
}


// Runs full training of network to n iterations. Takes ownership of Network (Not a borrow)
pub fn training_loop(epochs: usize, path: String, mut network: Network, min: &f64, max: &f64, scaled_input_matrix: &Matrix, scaled_expected_matrix: &Matrix) {
    let mut inputs   = scaled_input_matrix.clone();
    let expected     = scaled_expected_matrix.clone(); 
    let initial_learning_rate = 1e-7;

    // Loop over all epochs.
    for epoch in 1..=epochs {
        let learning_rate = exp_decay_learning_rate(initial_learning_rate, epoch as f64, 0.90);
        // Get expected output from input 
        let output = network.forward(&mut inputs);
        // output.print();
        let loss_matrix = output.sub(&expected).unwrap();
        // Collect loss
        let loss = loss_matrix.data.iter()
            .map(|x| x.powi(2)).sum::<f64>() / (output.row * output.col) as f64;

        // Backpropagate error 
        network.backward(&expected, learning_rate);

        println!("Epoch {} Loss {}", epoch, loss);

        // Save epoch data to file 
        let unscaled_output = reverse_normalize(&output, min, max, true);
        let _ = save_epoch_output(&epoch, &unscaled_output.data);
    }
    let _ = save_weights(&network, &path);
}

// Pass network through by ownership
//
// This function is hardcoded for input size etc since that relates to goal of use 
pub fn predict_output(mut network: Network) -> io::Result<()> {
    // Create hardcoded input signal
    let step: f64   = (10.0) / (1000 as f64);
    let mut dx: f64 = 0.0;
    let mut old_inputs: Vec<f64> = Vec::new();
    let mut new_inputs: Vec<f64> = Vec::new();

    for _ in 0..1000 {
        old_inputs.push(dx);
        dx += step;
    }
    for _ in 1000..2000 {
        new_inputs.push(dx);
        dx += step;
    }

    let mut old_input_matrix = Matrix::new(0.0, 1000, 1);
    let mut new_input_matrix = Matrix::new(0.0, 1000, 1);
    old_input_matrix.fill(&old_inputs);
    new_input_matrix.fill(&new_inputs);

    old_input_matrix = normalize_input(&old_input_matrix, true);
    new_input_matrix = normalize_input(&new_input_matrix, true);

    let old_outputs = network.forward(&mut old_input_matrix);
    let new_outputs = network.forward(&mut new_input_matrix);
    
    // Take one forward pass and fprintf original inputs and new outputs to file 
    let old_outputs = reverse_normalize(&old_outputs, &-1.0, &1.0, true); 
    let new_outputs = reverse_normalize(&new_outputs, &-1.0, &1.0, true);

    let mut inputs  = vec![0.0; 2000];
    let mut outputs = vec![0.0; 2000];
    for i in 0..2000 {
        if i < 1000 {
            inputs[i]  = old_inputs[i];
            outputs[i] = old_outputs.data[i];
        } else {
            inputs[i]  = new_inputs[i - 1000];
            outputs[i] = new_outputs.data[i - 1000];
        }
    }

    let mut file = File::create("prediction/output.txt")?;
    for i in 0..inputs.len() {
        writeln!(file, "{},{}", inputs[i], outputs[i])?;
    }
    
    Ok(())
}

pub fn get_min_max(matrix: &Matrix) -> (f64, f64) {
    let mut min = matrix.data[0];
    let mut max = matrix.data[0];

    for &value in matrix.data.iter() {
        if value < min {
            min = value;
        } else if value > max {
            max = value;
        }
    }
    
    (min, max)
}

// Normalize data to [-1, 1]
pub fn normalize_input(input_matrix: &Matrix, neg: bool) -> Matrix {
    let (min, max) = get_min_max(&input_matrix);
    
    let mut scaled_inputs = Matrix::new(0.0, input_matrix.row, input_matrix.col);
    for i in 0..input_matrix.row {
        for j in 0..input_matrix.col {
            let value = input_matrix.data[i * input_matrix.col + j];
            // Find the indexed value scaled to normalization range 
            let scale_value: f64 = if neg {
                2.0 * (value - min) / (max - min) - 1.0
            } else {
                (value - min) / (max - min)
            };

            scaled_inputs.data[i * input_matrix.col + j] = scale_value; 
        }
    }
    scaled_inputs
}

pub fn reverse_normalize(scaled_input: &Matrix, min: &f64, max: &f64, neg: bool) -> Matrix {
    let mut unscaled = Matrix::new(0.0, scaled_input.row, scaled_input.col);
    for i in 0..scaled_input.row {
        for j in 0..scaled_input.col {
            let scaled_value = scaled_input.data[i * scaled_input.col + j];

            unscaled.data[i * scaled_input.col + j] = if neg {
                (scaled_value + 1.0) * (max - min) / 2.0 + min
            } else {
                scaled_value * (max - min) + min
            };
        }
    }
    unscaled
}

fn main() -> io::Result<()> {
    // Link input and output count into neuron count
    // S.t a cmd line input [1,32,16,8,1] means SISO 1x32, 32x16, 16x8 network

    let args: Vec<String> = env::args().collect();

    // Expected inputs:
    //   cargo run data.txt [a,b,c,etc] -t          -- 3 args
    //   cargo run weights.txt -p                   -- 2 args 

    match args.len() {
        3 => {
            let io_type = &args[2];
            if io_type != "-p" { 
                panic!("Invalid! Usage: [cargo run] [data/weights.txt] [-t/-p/-l] [counts (if -t)]"); 
            }

            // Call to prediction after inputs are checked as accurate
            let file_arg = &args[1];
            let file   = File::open(file_arg)?;
            let reader = BufReader::new(file);
            let mut lines = reader.lines();

            let pulled_counts_line = lines.next().unwrap()?;
            let pulled_counts_str  = pulled_counts_line.trim_matches(|chars| chars == '[' || chars == ']');
            let pulled_counts: Result<Vec<usize>, _> = pulled_counts_str
                .split(',')
                .map(|x| x.trim().parse::<usize>())
                .collect();
            
            let pulled_counts = pulled_counts.unwrap();
            let counts = pulled_counts.clone();
            println!("{:?}", counts);

            let mut functions: Vec<Activation> = Vec::new();
            for i in 0..counts.len() {
                if i == counts.len() - 1 {
                    functions.push(identity_function());
                } else {
                    functions.push(tanh_function());
                }
            }

            let mut network = Network::new(&counts, functions);
            println!("Network: {:?}", network.sizes); 

            for (i, layer) in network.layers.iter_mut().enumerate() {

                for j in 0..counts[i + 1] {
                    for k in 0..counts[i] {
                        let weight_line = lines.next().unwrap()?;
                        let weight = weight_line.trim().parse::<f64>().unwrap();
                        layer.weights.data[j * counts[i] + k] = weight;
                    }
                }

                for j in 0..counts[i+1] {
                    let bias_line = lines.next().unwrap()?;
                    let bias = bias_line.trim().parse::<f64>().unwrap();
                    layer.biases.data[j] = bias;
                }
            }

            let _ = predict_output(network);
        },
        4 => {
            let io_type = &args[2];
            if io_type == "-l" {
                let file_arg   = &args[1];
                let weight_arg = &args[3];
                
                let file   = File::open(weight_arg)?;
                let reader = BufReader::new(file);
                let mut lines = reader.lines();

                let pulled_counts_line = lines.next().unwrap()?;
                let pulled_counts_str  = pulled_counts_line.trim_matches(|chars| chars == '[' || chars == ']');
                let pulled_counts: Result<Vec<usize>, _> = pulled_counts_str
                    .split(',')
                    .map(|x| x.trim().parse::<usize>())
                    .collect();
                
                let pulled_counts = pulled_counts.unwrap();
                let counts = pulled_counts.clone();
                println!("{:?}", counts);

                let mut functions: Vec<Activation> = Vec::new();
                for i in 0..counts.len() {
                    if i == counts.len() - 1 {
                        functions.push(identity_function());
                    } else {
                        functions.push(tanh_function());
                    }
                }

                let mut network = Network::new(&counts, functions);
                println!("Network: {:?}", network.sizes); 

                for (i, layer) in network.layers.iter_mut().enumerate() {

                    for j in 0..counts[i + 1] {
                        for k in 0..counts[i] {
                            let weight_line = lines.next().unwrap()?;
                            let weight = weight_line.trim().parse::<f64>().unwrap();
                            layer.weights.data[j * counts[i] + k] = weight;
                        }
                    }

                    for j in 0..counts[i+1] {
                        let bias_line = lines.next().unwrap()?;
                        let bias = bias_line.trim().parse::<f64>().unwrap();
                        layer.biases.data[j] = bias;
                    }
                }

                let file   = File::open(file_arg)?;
                let reader = BufReader::new(file);
                let mut data: Vec<Vec<f64>> = Vec::new();
                // Fill data array with all values on one row 
                for line in reader.lines() {
                    let line = line?;
                    let line = line.trim();

                    // Collect all values from the trimmed line 
                    let values: Vec<f64> = line
                        .split(',')
                        .map(|x| x.parse::<f64>().unwrap())
                        .collect();
                    
                    // Push values into array
                    data.push(values); 
                }
                // println!("{:?}", data);

                // Filter outputs and inputs into arrays 
                let input_count  = counts[0];
                let output_count = counts.last().unwrap();

                println!("Input Count {input_count}, Output Count {output_count}, Data Len {}", data.len());
                let (mut inputs_vec, mut outputs_vec): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
                // Fill data into two flattened arrays 
                for i in 0..data.len() {
                    // Read all inputs into correct array
                    for j in 0..input_count {
                        inputs_vec.push(data[i][j]);
                    }

                    // Read all expected values starting from input_count index
                    for k in input_count..(output_count + input_count) {
                        outputs_vec.push(data[i][k]);
                    }
                }
                
                let mut inputs: Matrix  = Matrix::new(0.0, data.len(), input_count);
                let mut outputs: Matrix = Matrix::new(0.0, data.len(), *output_count); 

                // Create matrices from data
                inputs.fill(&inputs_vec);
                outputs.fill(&outputs_vec);

                // Scale to range -1 to 1 
                let scaled_inputs = normalize_input(&inputs, true);
                let scaled_expected = normalize_input(&outputs, true);
                let (omin, omax) = get_min_max(&outputs);
                // Create network and call training loop 
                let path = "learned_weights.txt".to_string();
                training_loop(400, path, network, &omin, &omax, &scaled_inputs, &scaled_expected);

            } else if io_type != "-t" {
                panic!("Invalid! Usage: [cargo run] [data/weights.txt] [-t/p] [counts (if -t)]");
            } else {

                // Call to training loop after inputs are checked and network is created 
                let file_arg = &args[1];
                let count_arg = &args[3];

                // Parse counts and clone
                let counts = parse_counts(&count_arg).unwrap();
                let network_size = counts.clone();
                let mut functions: Vec<Activation> = Vec::new();
                for i in 0..counts.len() {
                    if i == counts.len() - 1 {
                        functions.push(identity_function());
                    } else {
                        functions.push(arctan_function());
                    }
                }

                let file   = File::open(file_arg)?;
                let reader = BufReader::new(file);

                // Fetch io counts from counts array
                let input_count  = counts[0];
                let output_count = counts.into_iter().last().unwrap(); 

                let mut data: Vec<Vec<f64>> = Vec::new();
                // Fill data array with all values on one row 
                for line in reader.lines() {
                    let line = line?;
                    let line = line.trim();

                    // Collect all values from the trimmed line 
                    let values: Vec<f64> = line
                        .split(',')
                        .map(|x| x.parse::<f64>().unwrap())
                        .collect();
                    
                    // Push values into array
                    data.push(values); 
                }
                // println!("{:?}", data);

                // Filter outputs and inputs into arrays 
                println!("Input Count {input_count}, Output Count {output_count}, Data Len {}", data.len());
                let (mut inputs_vec, mut outputs_vec): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());
                // Fill data into two flattened arrays 
                for i in 0..data.len() {
                    // Read all inputs into correct array
                    for j in 0..input_count {
                        inputs_vec.push(data[i][j]);
                    }

                    // Read all expected values starting from input_count index
                    for k in input_count..(output_count + input_count) {
                        outputs_vec.push(data[i][k]);
                    }
                }
                
                let mut inputs: Matrix  = Matrix::new(0.0, data.len(), input_count);
                let mut outputs: Matrix = Matrix::new(0.0, data.len(), output_count); 

                // Create matrices from data
                inputs.fill(&inputs_vec);
                outputs.fill(&outputs_vec);

                // Scale to range -1 to 1 
                let scaled_inputs = normalize_input(&inputs, true);
                let scaled_expected = normalize_input(&outputs, true);
                let (omin, omax) = get_min_max(&outputs);
                // Create network and call training loop 
                let network = Network::new(&network_size, functions);
                println!("{:?}", network.sizes);
                let path = "training_weights.txt".to_string();
                training_loop(2000, path, network, &omin, &omax, &scaled_inputs, &scaled_expected);
            }
        },
        _ => {
            panic!("Invalid! Usage: [cargo run] [data/weights.txt] [-t/p] [counts (if -t)]");
        }
    }

    Ok(()) 
}
