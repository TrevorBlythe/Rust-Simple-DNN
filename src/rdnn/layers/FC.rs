use crate::rdnn::layers::GenericLayer::GenericLayer;
use rand::Rng;

pub struct FC {
    pub w: Vec<f32>,  //weights
    pub ws: Vec<f32>, //weight's sensitivites
    pub b: Vec<f32>,  //bias
    pub bs: Vec<f32>, //bias sensitivities
    // pub in_data:Box<Vec<f32>>,
    pub out_data: Vec<f32>, //the "in data" for the next layer
    pub costs: Vec<f32>,    //error of each "neuron/node/in data"
    pub in_size: usize,
    pub out_size: usize,
}

impl GenericLayer for FC {
    fn get_name(&self) -> String {
        return String::from("Fc");
    }

    fn get_in_size(&self) -> usize {
        self.in_size
    }

    fn get_out_size(&self) -> usize {
        self.out_size
    }
    fn get_params_and_grads(&mut self) -> (&mut Vec<f32>, &mut Vec<f32>) {
        (&mut self.w, &mut self.ws)
    }

    fn get_weights_mut(&mut self) -> &mut Vec<f32> {
        &mut self.w
    }

    fn get_grads(&self) -> &Vec<f32> {
        &self.ws
    }

    fn get_costs(&self) -> &Vec<f32> {
        &self.costs
    }

    fn get_out_data(&self) -> &Vec<f32> {
        &self.out_data
    }

    fn get_weights(&self) -> &Vec<f32> {
        &self.out_data
    }

    fn is_trainable(&self) -> bool {
        return true;
    }

    fn backward_data(&mut self, data_in: &Vec<f32>, expected: &Vec<f32>) {
        for i in &mut self.costs {
            //reset the costs
            *i = 0.0;
        }
        for i in 0..=self.out_size - 1 {
            let err = expected[i] - self.out_data[i];
            for j in 0..=self.in_size - 1 {
                //activation times error = change to the weight to reduce error
                // weight times error = change to the activation to reduce error (cost)
                self.ws[i + j * self.out_size] += data_in[j] * err * 2.;
                self.costs[j] += self.w[i + j * self.out_size] * err * 2.;
            }
            self.bs[i] += err;
            //bias grad is real simple :)
        }

        for j in 0..=self.in_size - 1 {
            self.costs[j] /= self.out_size as f32; //finish averaging out the costs
        }
    }

    fn backward_costs(&mut self, data_in: &Vec<f32>, costs: &Vec<f32>) {
        for i in &mut self.costs {
            //reset the costs
            *i = 0.0;
        }
        for i in 0..=self.out_size - 1 {
            for j in 0..=self.in_size - 1 {
                //activation times error = change to the weight to reduce error
                // weight times error = change to the activation to reduce error (cost)
                self.ws[i + j * self.out_size] += data_in[j] * costs[i] * 2.;
                self.costs[j] += self.w[i + j * self.out_size] * costs[i] * 2.;
            }
            self.bs[i] += costs[i];
            //bias grad is real simple :)
        }

        for j in 0..=self.in_size - 1 {
            self.costs[j] /= self.out_size as f32; //finish averaging out the costs
        }
    }

    fn forward_data(&mut self, data: &Vec<f32>) {
        for i in 0..=self.out_size - 1 {
            self.out_data[i] = 0.0;
            for j in 0..=self.in_size - 1 {
                self.out_data[i] += data[j] * self.w[i + j * self.out_size];
                //this is the dirty deed of AI
            }
            self.out_data[i] += self.b[i]; // add bias :)
        }
    }
}

pub fn new(in_size: usize, out_size: usize) -> Box<FC> {
    return Box::new(FC {
        w: (0..in_size * out_size)
            .map(|_| {
                    rand::rng().random::<f32>()
                    * if rand::rng().random_bool(0.5) {
                        1.0
                    } else {
                        -1.0
                    }
            })
            .collect(),
        b: (0..out_size)
            .map(|_| {
                    rand::rng().random::<f32>()
                    * if rand::rng().random_bool(0.5) {
                        1.0
                    } else {
                        -1.0
                    }
            })
            .collect(),
        //vec of random floats (0-1) each one multiplied by a -1 or 1 (50% chance either). https://stackoverflow.com/questions/48218459/how-do-i-generate-a-vector-of-random-numbers-in-a-range
        ws: vec![0.0; in_size * out_size],
        bs: vec![0.0; out_size],
        costs: vec![0.0; in_size],
        out_data: vec![0.0; out_size],
        in_size: in_size,
        out_size: out_size,
    }); // return a boxed FC layer
}
