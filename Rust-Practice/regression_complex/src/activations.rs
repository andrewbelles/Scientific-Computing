use std::f64::consts::PI;

#[derive(Clone, PartialEq)]
pub enum ActivationType {
    Identity,
    Tanh,
    ArcTan,
    Sin,
    LeakyReLu,
    ReLu,
    ELU,
    Sigmoid,
    ShiftedSigmoid,
}

#[derive(Clone, PartialEq)]
pub struct Activation {
    pub function: fn(f64) -> f64,
    pub derivative: fn(f64) -> f64,
    pub activation_type: ActivationType 
}

impl Activation {
    pub fn new(
        function: fn(f64) -> f64,
        derivative: fn(f64) -> f64,
        activation_type: ActivationType
    ) -> Self {
        Activation { function, derivative, activation_type }
    }
}

pub fn identity_function() -> Activation {
    Activation::new(
        |x| x,
        |_x| 1.0,
        ActivationType::Identity
    )
}

#[allow(dead_code)]
pub fn tanh_function() -> Activation {
    Activation::new(
        |x| x.tanh(),
        |x| 1.0 - x.tanh().powi(2),
        ActivationType::Tanh,
    )
}

#[allow(dead_code)]
pub fn arctan_function() -> Activation {
    Activation::new(
        |x| (2.0/PI) * x.atan(),
        |x| 1.0 / (1.0 + x.powi(2)),
        ActivationType::ArcTan,
    )
}

#[allow(dead_code)]
pub fn sin_function() -> Activation {
    Activation::new(
        |x| x.sin(),
        |x| x.cos(),
        ActivationType::Sin,
    )
}

#[allow(dead_code)]
pub fn leakyrelu_function() -> Activation {
    Activation::new(
        |x| {
            if x > 0.0 {
                x
            } else {
                0.01 * x
            }
        },
        |x| {
            if x > 0.0 {
                1.0 
            } else {
                0.01
            }
        },
        ActivationType::LeakyReLu,
    )
}

#[allow(dead_code)]
pub fn relu_function() -> Activation {
    Activation::new(
        |x| {
            if x > 0.0 {
                x
            } else {
                0.0
            }
        },
        |x| {
            if x > 0.0 {
                1.0 
            } else {
                0.0
            }
        },
        ActivationType::ReLu,
    )
}

#[allow(dead_code)]
pub fn elu_function() -> Activation {
    Activation::new(
        |x| {
            if x > 0.0 {
                x
            } else {
                0.01 * (x.exp() - 1.0)
            }
        },
        |x| {
            if x > 0.0 {
                1.0 
            } else {
                0.01 * x.exp()
            }
        },
        ActivationType::ELU,
    )
}

#[allow(dead_code)]
pub fn sigmoid_function() -> Activation {
    Activation::new(
        |x| 1.0 / (1.0 + (-x).exp()),
        |x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            sigmoid * (1.0 - sigmoid)
        },
        ActivationType::Sigmoid,
    )
}

#[allow(dead_code)]
pub fn shifted_sigmoid_function() -> Activation {
    Activation::new(
        |x| 2.0 / (1.0 + (-x).exp()) - 1.0,
        |x| {
            let sigmoid = 1.0 / (1.0 + (-x).exp());
            2.0 * sigmoid * (1.0 - sigmoid)
        },
        ActivationType::ShiftedSigmoid
    )
}
