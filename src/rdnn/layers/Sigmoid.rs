use super::GenericLayerTrait::GenericLayer;

/*
Represents a layer in a neural network that performs the Sigmoid activation function
*/
pub struct Sigmoid {
    //empty params in this case, only exists to have something to return in get_param
    pub params: Vec<f32>, 
    pub grads: Vec<f32>, //same here ^^^
    pub out_data: Vec<f32>,
    pub input_grads: Vec<f32>,
    size: usize,
}

pub fn new(size: usize) -> Box<Sigmoid> {
    Box::new(Sigmoid {
        params: vec![0.0; 0], //has no params
        grads: vec![0.0; 0], //has no params, and therefore no grads
        out_data: vec![0.0; size],
        input_grads: vec![0.0; size],
        size,
    })
}


impl GenericLayer for Sigmoid {
    fn forward_data(&mut self, data: &Vec<f32>) {
        for i in 0..self.size {
            self.out_data[i] = 1.0 / (1.0 + (-data[i]).exp());
        }
    }

    fn backward_target(&mut self, _data_in: &Vec<f32>, expected: &Vec<f32>) {
        for i in 0..self.size {
            let sigmoid_derivative = self.out_data[i] * (1.0 - self.out_data[i]);
            self.input_grads[i] = (expected[i] - self.out_data[i]) * sigmoid_derivative;
        }
    }

    fn backward_grads(&mut self, _data_in: &Vec<f32>, grads: &Vec<f32>) {
        for i in 0..self.size {
            let sigmoid_derivative = self.out_data[i] * (1.0 - self.out_data[i]);
            self.input_grads[i] = grads[i] * sigmoid_derivative;
        }
    }

    fn get_out_data(&self) -> &Vec<f32> {
        &self.out_data
    }

    fn get_input_grads(&self) -> &Vec<f32> {
        &self.input_grads
    }

    fn get_name(&self) -> &str {
        "Sigmoid"
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
        vec![(&mut self.params,&mut self.grads)]
    }

    fn get_params_mut(&mut self) -> Vec<&mut Vec<f32>> {
        vec![&mut self.params] //in this layer, they are empty
    }

    fn get_grads(&self) -> Vec<&Vec<f32>> {
        vec![&self.grads] //in this layer, they are empty
    }

    fn get_params(&self) -> Vec<&Vec<f32>> {
        vec![&self.params] //in this layer, they are empty
    }

}