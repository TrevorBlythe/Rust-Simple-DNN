use crate::rdnn::layers::GenericLayer::GenericLayer;

//Sometimes Relu can make networks explode
pub struct Relu {
    pub out_data: Vec<f32>, //the input data for the next layer
    pub costs: Vec<f32>, //costs/error for each "neuron/node/input data" cost[0] is for last layers out_data[0]
    pub in_size: usize,
    pub out_size: usize,
}

pub fn new(size: usize) -> Box<Relu> {
    return Box::new(Relu {
        costs: vec![0.0; size],
        out_data: vec![0.0; size],
        in_size: size,
        out_size: size,
    }); // return a boxed Sig layer
}

impl GenericLayer for Relu {
    fn is_trainable(&self) -> bool {
        return false;
    }

    fn get_name(&self) -> String {
        return String::from("Relu");
    }

    fn get_in_size(&self) -> usize {
        self.in_size
    }

    fn get_out_size(&self) -> usize {
        self.out_size
    }
    fn get_params_and_grads(&mut self) -> (&mut Vec<f32>, &mut Vec<f32>) {
        panic!("shoudnt ever call this on a relu layer buddy");
    }

    fn get_weights_mut(&mut self) -> &mut Vec<f32> {
        panic!("nope");
    }

    fn get_grads(&self) -> &Vec<f32> {
        panic!("nope");
    }

    fn get_costs(&self) -> &Vec<f32> {
        &self.costs
    }

    fn get_out_data(&self) -> &Vec<f32> {
        &self.out_data
    }

    fn get_weights(&self) -> &Vec<f32> {
        panic!("nope");
    }

    fn backward_data(&mut self, _data_in: &Vec<f32>, expected: &Vec<f32>) {
        for j in 0..self.in_size {
            if self.out_data[j] > 0.0{
                let z = expected[j] - self.out_data[j];
                self.costs[j] = z;
            }
        }
    }

    fn backward_costs(&mut self, _data_in: &Vec<f32>, costs: &Vec<f32>) {
        for j in 0..self.in_size {
            if self.out_data[j] > 0.0{
                self.costs[j] = costs[j];
            }
        }
    }

    fn forward_data(&mut self, data: &Vec<f32>) {
        for i in 0..self.out_data.len() {
            if data[i] > 0.0 {
                self.out_data[i] = data[i];
            }else{
                self.out_data[i] = 0.0;
            }
        }
    }
}
