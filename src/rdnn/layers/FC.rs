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
        for i in &mut self.input_grads {
            //reset the input grads
            *i = 0.0;
        }
        for i in 0..=self.out_size - 1 {
            let err = expected[i] - self.out_data[i];
            for j in 0..=self.in_size - 1 {
                //activation times error = change to the weight to reduce error
                // weight times error = change to the activation to reduce error (cost)
                self.weightsGrads[i + j * self.out_size] += data_in[j] * err * 2.;
                self.input_grads[j] += self.weights[i + j * self.out_size] * err * 2.;
            }
            self.biasGrads[i] += err;
            //bias grad is real simple :)
        }

        for j in 0..=self.in_size - 1 {
            self.input_grads[j] /= self.out_size as f32; //finish averaging out the costs
        }
    }

    fn backward_grads(&mut self, data_in: &Vec<f32>, grads: &Vec<f32>) {
        for i in &mut self.input_grads {
            //reset the input grads
            *i = 0.0;
        }
        for i in 0..=self.out_size - 1 {
            for j in 0..=self.in_size - 1 {
                //activation times error = change to the weight to reduce error
                // weight times error = change to the activation to reduce error (cost)
                self.weightsGrads[i + j * self.out_size] += data_in[j] * grads[i] * 2.;
                self.input_grads[j] += self.weights[i + j * self.out_size] * grads[i] * 2.;
            }
            self.biasGrads[i] += grads[i];
            //bias grad is real simple :)
        }

        for j in 0..=self.in_size - 1 {
            self.input_grads[j] /= self.out_size as f32; //finish averaging out the costs
        }
    }

    fn forward_data(&mut self, data: &Vec<f32>) {
        for i in 0..=self.out_size - 1 {
            self.out_data[i] = 0.0;
            for j in 0..=self.in_size - 1 {
                self.out_data[i] += data[j] * self.weights[i + j * self.out_size];
                //this is the dirty deed of AI
            }
            self.out_data[i] += self.bias[i]; // add bias :)
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
