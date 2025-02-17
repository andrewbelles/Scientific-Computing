use std::fs::File; 
use std::io::{self, BufReader, BufRead, Write};
use std::env;
use rand::Rng;
use rand_distr::{Normal, Distribution};

// Simple aggregate holder of variance, mean, and count  
#[derive(Copy, Clone, Debug)]
pub struct Aggregate {
    mean:  f64,
    m2:    f64,
    count: u64
}

// Generate large quantities of random test data to write to file.
pub fn generate_data(size: u64) -> Vec<f64> {
    let mut data: Vec<f64> = Vec::new();
    let mut rng = rand::thread_rng();
    let scale: f64  = rng.gen_range(2.0..3.0);
    let offset: f64 = rng.gen_range(40.0..100.0); 

    let normal = Normal::new(0.5 * scale + offset, scale).unwrap(); 

    for _ in 0..size {
        // create value 
        let value: f64 = normal.sample(&mut rng);
        data.push(value);
    } 

    return data;
}


// Implementation of welfords online algorithm
// Calculates the variance and mean through a running count type approximation 
pub fn aggregate_stdev(current_value: &f64, current_aggregate: &mut Aggregate) {

    // Increment aggregate count
    current_aggregate.count += 1;

    // Calculate delta, update mean, and find delta again
    let delta: f64 = current_value - current_aggregate.mean;
    current_aggregate.mean += delta / (current_aggregate.count as f64);
    let delta_2: f64 = current_value - current_aggregate.mean;

    // update variance 
    current_aggregate.m2 += delta * delta_2;
}

pub fn string_to_int(argument: &String) -> Option<u64> {
    if argument.starts_with("size=") {
        let slice   = &argument[5..];
        
        match slice.trim().parse::<u64>() {
            Ok(size) => Some(size),
            Err(e) => {
                eprintln!("Parse error {e}");
                None
            }
        }
    } else {
        None
    }
}

fn main() -> io::Result<()> {
    let args: Vec<String> = env::args().collect();

    match args.len() {
        4 if args[1] == "gen" => {
            let size_argument = &args[2];
            let output_path   = &args[3];
            

            if let Some(generate_size) = string_to_int(size_argument) {

                println!("Switching Process... Generating Large Test File");
                // Generate array of data 
                let data = generate_data(generate_size);
                let mut file = File::create(output_path)?;
                let values = &data[..];
                for value in values.iter() {
                    writeln!(&mut file, "{:e}", value)?;
                }
            } else {
                eprintln!("Invalid Size argument\nUsage: [cargo run] [gen] [size=n] [test_data.txt]");
            }
        }
        _ => {
            if args.len() != 3 {
                println!("Invalid Argument Count: Usage [cargo run] [datafile.txt] [output.txt]");
                println!("  >> Alternative Usage: [cargo run] [-g] [-size=n] [output.txt]");
                return Ok(());

            }    
        }
    }

    let input_path  = &args[1];
    let output_path = &args[2];

    let file = File::open(input_path)?;
    let reader = BufReader::new(file);
    let mut x: Vec<f64> = Vec::new();
    let mut aggregates = Aggregate {
        m2:    0.0,
        mean:  0.0,
        count: 0
    };

    // Reads each value, stores it in x and calculates the standard deviation/mean for data set 
    for line in reader.lines() {
        let line = line?;
        let line = line.trim(); 

        // Check for parse error and place value in x vec
        match line.parse::<f64>() {
            Ok(read_value) => {
                x.push(read_value);
                // update aggregate 
                aggregate_stdev(&read_value, &mut aggregates);
            }
            Err(e) => {
                eprintln!("Parse Error (line, e) ({}, {})", line, e);
            }
        }
    }

    // Find standard deviation
    let mut stdev: f64 = (aggregates.m2 / ((aggregates.count - 1) as f64)).sqrt();

    println!(">> Initial Dataset: ");
    println!("    Stdev: {stdev}");
    println!("    Mean : {}", aggregates.mean);
    println!("    Count: {}", aggregates.count);

    println!(">> Enter an exclusion threshold: ");
    
    // Parse input string 

    let mut threshold_str = String::new();
    io::stdin()
        .read_line(&mut threshold_str)
        .expect("Failed to read line");

    let tr_threshold = threshold_str.trim();
    let threshold: f64 = match tr_threshold.parse::<f64>() {
        Ok(value) => value,
        Err(e) => {
            eprintln!("Parse Error {}", e);
            return Ok(());
        }
    };

    println!("\n>> Threshold: {}", threshold);

    let mut current_aggregates = Aggregate {
        mean: 0.0,
        m2:   0.0,
        count:  0
    };
    let mut previous_aggregates = aggregates;

    // Create "sieve vectors"
    let (mut c, mut o): (Vec<f64>, Vec<f64>) = (Vec::new(), Vec::new());

    // Slice x into values 
    let mut values = &mut x[..];
    let mut iteration = 0;
    loop {
        let mut outlier_count: u64 = 0; 
        // Sieves value into either clean or outlier vectors
        for value in values.iter_mut() {
            // If value is an outlier 
            if (*value - previous_aggregates.mean).abs() > stdev * threshold {
                outlier_count += 1;
                o.push(*value);
            } else {    // Clean value
                aggregate_stdev(value, &mut current_aggregates);
                c.push(*value);
            }
        }

        if current_aggregates.count < 2 {
            println!("Threshold Too Restrictive!");
            return Ok(());
        }


        // Update standard deviation 
        stdev = (current_aggregates.m2 / (current_aggregates.count as f64)).sqrt();
        x = c;
        values = &mut x[..];
        c = Vec::new();

        // Update aggregates and counts
        previous_aggregates    = current_aggregates;
        iteration += 1;
        
        // Print iteration results or end if no outliers were found 
        if outlier_count != 0 {
            println!(">> Iteration {iteration}");
            println!("    Stdev: {stdev}");
            println!("    Mean : {}", current_aggregates.mean);
            println!("    Outlier Count: {}", outlier_count);
        } else {
            break;
        }
    }

    let mut file = File::create(output_path)?;
    let values = &o[..];
    for value in values.iter() {
        writeln!(&mut file, "{:e}", value)?;
    }

    println!(">> Final Dataset: ");
    println!("    Stdev: {stdev}");
    println!("    Mean : {}", current_aggregates.mean);
    println!("    Outlier Count: {}", o.len());

    Ok(())
}
