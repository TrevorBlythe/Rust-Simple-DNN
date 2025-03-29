use super::GenericLayerTrait::GenericLayer;
use rand::Rng;
/*
Represents a layer in a neural network that performs Linear/Dense/Fully Connected operation

THIS IS A GOOD TEMPLATE IF YOU WANT TO WRITE YOUR OWN LAYER FOR CUSTOM FUNCTIONALITY. 
All you have to do is change the forward function to do whatever data processing you like, 
and then of course write the backwards function to store grads somewhere. 
3blue1browns videos on neural nets will make you understand it all 
*/
pub struct FC {
    pub weights: Vec<f32>,  //weights
    pub weightsGrads: Vec<f32>, //weight's sensitivites
    pub bias: Vec<f32>,  //bias
    pub biasGrads: Vec<f32>, //bias sensitivities
    pub out_data: Vec<f32>, //The output of this layer, this feild is the input for the next layer
    pub input_grads: Vec<f32>,    //The error of each input "neuron/node/data point"
    in_size: usize, // Not public, I want to be explicit that changing this field does nothing.
    out_size: usize, // Same here
}

impl GenericLayer for FC {
    fn get_name(&self) -> &str {
        return "FC";
    }

    fn get_input_grads(&self) -> &Vec<f32> {
        &self.input_grads //The error of each input "neuron/node/data point"
    }

    fn get_out_data(&self) -> &Vec<f32> {
        &self.out_data //The output of this layer, this feild is the input for the next layer
    }

    fn is_trainable(&self) -> bool {
        return true;
    }

    fn backward_target(&mut self, data_in: &Vec<f32>, expected: &Vec<f32>) {
        self.input_grads.iter_mut().for_each(|i| *i = 0.0);

            for (i, &expected_val) in expected.iter().enumerate() {
                let err = expected_val - self.out_data[i];
                for (j, &data_val) in data_in.iter().enumerate() {
                    self.weightsGrads[i + j * self.out_size] += data_val * err * 2.0;
                    self.input_grads[j] += self.weights[i + j * self.out_size] * err * 2.0;
                }
                self.biasGrads[i] += err;
            }

            self.input_grads.iter_mut().for_each(|input_grad| *input_grad /= self.out_size as f32);
    }

    fn backward_grads(&mut self, data_in: &Vec<f32>, grads: &Vec<f32>) {
        self.input_grads.iter_mut().for_each(|i| *i = 0.0);

        for (i, &grad) in grads.iter().enumerate() {
            for (j, &data_val) in data_in.iter().enumerate() {
                self.weightsGrads[i + j * self.out_size] += data_val * grad * 2.0;
                self.input_grads[j] += self.weights[i + j * self.out_size] * grad * 2.0;
            }
            self.biasGrads[i] += grad;
        }

        self.input_grads.iter_mut().for_each(|input_grad| *input_grad /= self.out_size as f32);
    }

    fn forward_data(&mut self, data: &Vec<f32>) {
        for (i, out_val) in self.out_data.iter_mut().enumerate() {
            *out_val = self.bias[i]; // Initialize with bias

            for (j, &data_val) in data.iter().enumerate() {
                *out_val += data_val * self.weights[i + j * self.out_size];
            }
        }
    }

    fn get_params_and_grads(&mut self) -> Vec<(&mut Vec<f32>, &mut Vec<f32>)> {
        vec![
        (&mut self.weights,&mut self.weightsGrads),
        (&mut self.bias, &mut self.biasGrads)
        ]
    }

    fn get_params_mut(&mut self) -> Vec<&mut Vec<f32>> {
        vec![&mut self.weights, &mut self.bias]
    }

    fn get_grads(&self) -> Vec<&Vec<f32>> {
        vec![&self.weightsGrads, &self.biasGrads]
    }

    fn get_params(&self) -> Vec<&Vec<f32>> {
        vec![&self.weights, &self.bias]
    }

    fn get_in_size(&self) -> usize {
        self.in_size
    }

    fn get_out_size(&self) -> usize {
        self.out_size
    }
}

pub fn new(in_size: usize, out_size: usize) -> Box<FC> {
    return Box::new(FC {
        weights: (0..in_size * out_size)
            .map(|_| rand::rng().random_range(-1.0..1.0))
            .collect::<Vec<f32>>(), //generates weights from -1, 1
        bias: (0..in_size * out_size)
            .map(|_| rand::rng().random_range(-1.0..1.0))
            .collect::<Vec<f32>>(), //generates biases from -1, 1

        weightsGrads: vec![0.0; in_size * out_size], //this will store gradients for the weights.
        biasGrads: vec![0.0; out_size],
        input_grads: vec![0.0; in_size],
        out_data: vec![0.0; out_size],
        in_size: in_size,
        out_size: out_size,
    }); // return a boxed FC layer
}
