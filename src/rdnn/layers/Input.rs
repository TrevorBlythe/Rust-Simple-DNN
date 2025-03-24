use crate::rdnn::layers::GenericLayer::GenericLayer;

pub struct Input {
    pub out_data: Vec<f32>, //the "in data" for the next layer
    pub in_size: usize,
    pub out_size: usize,
}

impl GenericLayer for Input {
    fn is_trainable(&self) -> bool {
        return false;
    }

    fn get_name(&self) -> String {
        return String::from("Input");
    }

    fn get_in_size(&self) -> usize {
        self.in_size
    }

    fn get_out_size(&self) -> usize {
        self.out_size
    }

    fn get_params_and_grads(&mut self) -> (&mut Vec<f32>, &mut Vec<f32>) {
        panic!("Get_params_and_grads was called somewhere in your code on an input layer. This should never happen");
    }

    fn get_weights_mut(&mut self) -> &mut Vec<f32> {
        panic!("get_weights_mut was ran on an input layer, something is wrong")
    }

    fn get_grads(&self) -> &Vec<f32> {
        panic!("nope");
    }

    fn get_costs(&self) -> &Vec<f32> {
        panic!("nope");
    }

    fn get_out_data(&self) -> &Vec<f32> {
        &self.out_data
    }

    fn get_weights(&self) -> &Vec<f32> {
        panic!("nope");
    }

    fn backward_data(&mut self, _data_in: &Vec<f32>, _expected: &Vec<f32>) {
        //does nothing lmao
        panic!("somehow backwards data was called on an input. something is wrong");
    }

    fn backward_costs(&mut self, _data_in: &Vec<f32>, _costs: &Vec<f32>) {
        panic!("somehow backwards costs was called on an input. something is wrong");
    }

    fn forward_data(&mut self, data: &Vec<f32>) {
        let mut x = 0;
        for i in data {
            self.out_data[x] = *i;
            x += 1;
        }
    }
}

pub fn new(in_size: usize) -> Box<Input> {
    return Box::new(Input {
        out_data: vec![0.0; in_size],
        in_size: in_size,
        out_size: in_size,
    });
}
