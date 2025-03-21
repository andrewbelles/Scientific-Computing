use std::fmt::{self, Debug, Display, Formatter};
use std::time::{Duration, Instant};

// Error class to store potentially complex error from baseline from benchmark
struct Error<E, R> {
    value: E,
    function: Box<dyn Fn(R, R) -> E> 
}

// Takes return type from benchmark and the type of the Error
impl<E, R> Error<E, R> 
where 
    R: Debug,
    E: Default + Debug,
{
    // Instantiates new error object for specific error function 
    fn new(func: impl Fn(R, R) -> E + 'static) -> Self {
        Self {
            value: Default::default(),
            function: Box::new(func),
        }
    }
    // Calculates the error and stores it in the value member
    fn get_error(&mut self, base: R, result: R) {
        self.value = (self.function)(base, result);
    }
}

// Implement Display for error to show error from baseline to user 
impl<E, R> Display for Error<E, R>
where 
    E: Display,
    R: Debug
{
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        write!(f, "{{ error value: {} }}", self.value)
    }
} 

// Benchmark class to take functions flexibly, compare to baseline, and compute some useful error
// along with runtime
struct Benchmark<R, E, Args> {
    bench_funcs: Vec<Box<dyn Fn(Args) -> R>>,
    results: Vec<(Duration, Error<E, R>)>,
    args: Args,
    iter: u32
}

impl<R, E, Args> Benchmark<R, E, Args>
where 
    R: Debug + Copy + 'static,
    E: Default + Debug + 'static,
    Args: Copy,
{
    fn new(iter: u32, base_func: Box<dyn Fn(Args) -> R>, error_func: Box<dyn Fn(R, R) -> E>, args: Args) -> Self {
        let mut b = Benchmark {
            bench_funcs: Vec::new(),
            results: Vec::new(), 
            args,
            iter
        };
        
        b.bench_funcs.push(base_func);
        // Push tuple (runtime, Error)
        b.results.push((Duration::default(), Error::<E, R>::new(error_func)));
        b.set_baseline();
        b
    }

    fn set_baseline(&mut self) {
        let mut runtime = Duration::default();

        for i in 0..(self.iter) {
            let start = Instant::now();

            if i == 0 {
                let baseline_result = (self.bench_funcs[0])(self.args);
                self.results[0].1.get_error(baseline_result, baseline_result);
            } else {
                let _ = (self.bench_funcs[0])(self.args);
            }
            runtime += start.elapsed();
        }
        let average_runtime = runtime / self.iter;
        self.results[0].0 = average_runtime;
    }

    fn insert(&mut self, func: impl Fn(Args) -> R + 'static) {
        self.bench_funcs.push(Box::new(func));
    }

    fn run(&mut self) {
        // Ensure there are more than one functions to benchmark
        assert!(self.bench_funcs.len() > 1);

        // Ensure there are enough initilized result tuples 
        if self.results.len() < self.bench_funcs.len() {
            while self.results.len() < self.bench_funcs.len() {
                self.results.push((Duration::default(), Error::<E, R>::new(self.results[0].1.function.clone())))
            }
        }
    }
}

fn main() {
   let mut error = Error::<f32, f32>::new(|x, y| x - y); 
   error.get_error(1.0, 0.99);
   println!("{error}")
}
