use super::GenericLayerTrait::GenericLayer;

pub struct LeakyRelu {
    pub params: Vec<f32>, 
    pub grads: Vec<f32>,
    pub out_data: Vec<f32>,
    pub input_grads: Vec<f32>,
    size: usize,
    alpha: f32, // Leaky factor
}

//alpha is the leak amount
pub fn new(size: usize, alpha: f32) -> Box<LeakyRelu> {
    Box::new(LeakyRelu {
        params: vec![0.0; 0],
        grads: vec![0.0; 0],
        out_data: vec![0.0; size],
        input_grads: vec![0.0; size],
        size,
        alpha,
    })
}

impl GenericLayer for LeakyRelu {
    fn forward_data(&mut self, data: &Vec<f32>) {
        for i in 0..self.size {
            self.out_data[i] = if data[i] > 0.0 { data[i] } else { self.alpha * data[i] };
        }
    }

    fn backward_target(&mut self, data_in: &Vec<f32>, expected: &Vec<f32>) {
        for i in 0..self.size {
            let derivative = if data_in[i] > 0.0 { 1.0 } else { self.alpha };
            self.input_grads[i] = (expected[i] - self.out_data[i]) * derivative;
        }
    }

    fn backward_grads(&mut self, data_in: &Vec<f32>, grads: &Vec<f32>) {
        for i in 0..self.size {
            let derivative = if data_in[i] > 0.0 { 1.0 } else { self.alpha };
            self.input_grads[i] = grads[i] * derivative;
        }
    }

    fn get_out_data(&self) -> &Vec<f32> {
        &self.out_data
    }

    fn get_input_grads(&self) -> &Vec<f32> {
        &self.input_grads
    }

    fn get_name(&self) -> &str {
        "LeakyReLU"
    }

    fn is_trainable(&self) -> bool {
        false
    }

    fn get_in_size(&self) -> usize {
        self.size
    }

    fn get_out_size(&self) -> usize {
        self.size
    }

    fn get_params_and_grads(&mut self) -> Vec<(&mut Vec<f32>, &mut Vec<f32>)> {
        vec![(&mut self.params, &mut self.grads)]
    }

    fn get_params_mut(&mut self) -> Vec<&mut Vec<f32>> {
        vec![&mut self.params]
    }

    fn get_grads(&self) -> Vec<&Vec<f32>> {
        vec![&self.grads]
    }

    fn get_params(&self) -> Vec<&Vec<f32>> {
        vec![&self.params]
    }
}
