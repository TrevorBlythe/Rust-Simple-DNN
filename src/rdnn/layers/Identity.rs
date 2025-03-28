use super::GenericLayerTrait::GenericLayer;

/*
Represents a layer in a neural network that does nothing.
Input data goes straight through unchanged, same with grads.
*/
pub struct Identity {
    //empty params in this case, only exists to have something to return in get_param
    pub params: Vec<f32>, 
    pub grads: Vec<f32>, //same here ^^^
    pub out_data: Vec<f32>,
    pub input_grads: Vec<f32>,
    size: usize,
}

pub fn new(size: usize) -> Box<Identity> {
    Box::new(Identity {
        params: vec![0.0; 0], //has no params
        grads: vec![0.0; 0], //has no params, and therefore no grads
        out_data: vec![0.0; size],
        input_grads: vec![0.0; size],
        size:size,
    })
}


impl GenericLayer for Identity {
    fn forward_data(&mut self, data: &Vec<f32>) {
        if self.out_data.len() != data.len(){
            panic!("Input size is wrong! Model expected size of: {}, but you inputted a size of: {}",self.out_data.len(), data.len());
        }
        self.out_data.copy_from_slice(data);
    }

    fn backward_target(&mut self, _data_in: &Vec<f32>, expected: &Vec<f32>) {
        // Pass the error directly as loss
        for i in 0..self.size {
            self.input_grads[i] = expected[i] - self.out_data[i];
        }
    }

    fn backward_grads(&mut self, _data_in: &Vec<f32>, grads: &Vec<f32>) {
        // Pass the losses directly
        self.input_grads.copy_from_slice(grads);
    }

    fn get_out_data(&self) -> &Vec<f32> {
        &self.out_data
    }

    fn get_input_grads(&self) -> &Vec<f32> {
        &self.input_grads
    }

    fn get_name(&self) -> &str {
        "Identity"
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
        vec![&mut self.params] //in this layer, they are empty, putting things in here will do nothing
    }

    fn get_grads(&self) -> Vec<&Vec<f32>> {
        vec![&self.grads] //in this layer, they are empty
    }

    fn get_params(&self) -> Vec<&Vec<f32>> {
        vec![&self.params] //in this layer, they are empty
    }

}