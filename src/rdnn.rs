// #![allow(dead_code)]
pub mod layers;

pub struct Net {
    //we use f32 cuz it works better with gpu im pretty sure (will do someday)
    pub layers: Vec<Box<dyn layers::GenericLayerTrait::GenericLayer>>, //a vector of layer structs each representing a layer in the AI model.
    layer_count: usize, //layers.len() basically.. Its only usize cuz thats whats returned from len()
    pub batch_size: i32,
    training_iterations: i32, //goes up +1 for every 'backwards' call to net.
    learning_rate: f32,
}

impl Net {
    pub fn new(
        mut layers: Vec<Box<dyn layers::GenericLayerTrait::GenericLayer>>,
        batch_size: i32,
        learning_rate: f32,
    ) -> Self {

        //Why do we need to put an input layer in every network?
        //because the structs that represent layers, dont store their inputs when you call forward(input),
        //yet it needs to know its inputs to train.
        //An input layer, doesnt need to train because it does absolutely nothing to the data.
        //So its input will always be its output. In this way it serves to store the inputs for the next layer.
        let first_layer_input_size = layers[0].get_in_size();
        layers.insert(0, layers::Identity::new(first_layer_input_size));

        //create the network struct.
        let net = Net {
            batch_size: batch_size,
            layer_count: layers.len(),
            layers: layers,
            training_iterations: 0,
            learning_rate: learning_rate,
        };
        return net;
    }

    pub fn print_layers(&self) {
        for i in &self.layers {
            println!("{:?}", i.get_name());
        }
    }

    pub fn forward_data(&mut self, data: &Vec<f32>) -> &Vec<f32> {
        self.layers[0].forward_data(data);
        for i in 1..self.layers.len() {
            //took me 30 mintues to figure out how to do this WHY WHY WHY!!!!
            //Basically in rust you can't modify something in a vector and also be accesssing another object
            //in the vector. But in my case I need to be able to see out_data from each layer's previous layer to forward it.
            //So i take the layer out which isnt to bad because its just a pointer to the actual object.
            let temp = self.layers.remove(i - 1);
            self.layers[i - 1].forward_data(temp.get_out_data());
            self.layers.insert(i - 1, temp);
        }
        //output the last layers output data
        self.layers[self.layers.len() - 1].get_out_data()
    }

    pub fn backward_data(&mut self, expected_output: &Vec<f32>) {
        let temp = self.layers.remove(self.layer_count - 2); //removes second from last layer (last layer is layercount-1)
        self.layers[self.layer_count - 2].backward_target(temp.get_out_data(), expected_output); //backwards the expected data into the real last layer. Note we are using the layer we took out to input the layers in_data for the function, (check params).
        self.layers.insert(self.layer_count - 2, temp);//puts it back

        for i in (1..=self.layers.len() - 2).rev() { //loops through all layers backwards except last and first ( we just did those )
            //remove the next and last layers.
            let temp = self.layers.remove(i + 1); 
            let temp_two = self.layers.remove(i - 1); 

            self.layers[i - 1].backward_grads(temp_two.get_out_data(), temp.get_input_grads());

            //reinsert the layers
            self.layers.insert(i - 1, temp_two);
            self.layers.insert(i + 1, temp);
        }

        //once batch size it reached then we add the gradients to teh weights.
        self.training_iterations += 1;
        if self.training_iterations % self.batch_size == 0 {
            let batch_size_as_f32: f32 = self.batch_size as f32;
            // Apply gradients to trainable layers
            for layer in self.layers.iter_mut().filter(|layer| layer.is_trainable()) {
                let param_grad_pairs = layer.get_params_and_grads();

                for (params, grads) in param_grad_pairs {
                    for (param, grad) in params.iter_mut().zip(grads.iter_mut()) {
                        // Apply gradient update
                        *param += (*grad / batch_size_as_f32) * self.learning_rate;

                        // Reset gradient
                        *grad = 0.0;
                    }
                }

            }
        }
    }
}


// SERDE SECTION BELOW (code for saving networks to files and loading networks) -------------
#[cfg(feature = "serde")]
use serde::{Serialize, Deserialize};
#[cfg(feature = "serde")]
use serde_json;
#[cfg(feature = "serde")]
use std::fs::File;
#[cfg(feature = "serde")]
use std::io::Read;
#[cfg(feature = "serde")]
use std::io::Write;

//This struct exists to hold the parameters of a network,
// and then get serialized/saved to a file with serde.
#[cfg(feature = "serde")]
#[derive(Serialize, Deserialize)]
struct SavedWeights {
    paramsForEachLayer: Vec<Vec<f32>>, // A Vector of each Layers Parameters
}

impl Net {
    //example: net.save_weights("network_weights.json");
    #[allow(unused_variables)] //it says path is unused cuz the feature gating
    pub fn save_weights(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        #[cfg(feature = "serde")]
        {
            let mut tempWeightsContainer: Vec<Vec<f32>> = Vec::new();
            for layer in &self.layers {
                for param in layer.get_params() {
                    tempWeightsContainer.push(param.clone());
                }
            }

            let data = serde_json::to_string(&SavedWeights { paramsForEachLayer: tempWeightsContainer }).unwrap();
            let mut file = File::create(path)?;
            file.write_all(data.as_bytes())?;

            Ok(()) 
        }
        #[cfg(not(feature = "serde"))]
        {
            eprintln!("rdnn error: serde feature not enabled, cannot save weights. HINT: use the serde feature. + Rust_Simple_DNN = {{ .. features= ['serde']}}");
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "rdnn error: serde feature not enabled, cannot save weights. HINT: use the serde feature. + Rust_Simple_DNN = { .. features= ['serde']}",
            )))
        }
    }

    #[allow(unused_variables)] //it says path is unused cuz the feature gating
    pub fn load_weights(&mut self, path: &str) -> Result<(), Box<dyn std::error::Error>>  {

        #[cfg(feature = "serde")]
        {
            let mut file = File::open(path)?;
            let mut data = String::new();
            file.read_to_string(&mut data)?;
            
            let saved: SavedWeights = serde_json::from_str(&data)?;
            let mut counter:usize = 0;
            for layer in self.layers.iter_mut() {
                let mut params_mut = layer.get_params_mut();
                
                // Iterate over the saved parameters and assign them to each parameter in the layer
                for params in params_mut.iter_mut() {
                    params.copy_from_slice(&saved.paramsForEachLayer[counter]);
                    counter += 1;
                }
            }
            Ok(())
        }
        #[cfg(not(feature = "serde"))]
        {
            eprintln!("rdnn error: serde feature not enabled, cannot save weights. HINT: use the serde feature. + Rust_Simple_DNN = {{ .. features= ['serde']}}");
            Err(Box::new(std::io::Error::new(
                std::io::ErrorKind::Other,
                "serde feature not enabled, cannot load weights HINT: use the serde feature. + Rust_Simple_DNN = { .. features= ['serde']}",
            )))
        }
    }

}
