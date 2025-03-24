#![allow(dead_code)]
pub mod layers;
pub struct Net {
    //we use f32 cuz it works better with gpu im pretty sure (will do someday)
    pub layers: Vec<Box<dyn layers::GenericLayer::GenericLayer>>, //a vector of layer structs each representing a layer in the AI model.
    layer_count: usize, //layers.len() basically.. Its only usize cuz thats whats returned from len()
    pub batch_size: i32,
    training_iterations: i32, //goes up +1 for every 'backwards' call to net.
    // pub even_layers:Vec<Box<dyn GenericLayer>>, //this is an idea for the future idk tho.
    // pub odd_layers:Vec<Box<dyn GenericLayer>>
    learning_rate: f32,
}

impl Net {
    pub fn new(
        mut layers: Vec<Box<dyn layers::GenericLayer::GenericLayer>>,
        batch_size: i32,
        learning_rate: f32,
    ) -> Self {

        //Why do we need to put an input layer in every network?
        //because layer structs dont store their inputs, therefore a trainable layer coudn't train because it woudnt know what its input was.
        //An input layer however, doesnt need to train because it does absolutely nothing to the data.
        //could be optimized probably by storing the first layer input data in the network struct instead and then we woudnt need it.
        let first_layer_input_size = layers[0].get_in_size();
        layers.insert(0, layers::Input::new(first_layer_input_size));

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
        self.layers[self.layer_count - 2].backward_data(temp.get_out_data(), expected_output); //backwards the expected data into the real last layer. Note we are using the layer we took out to input the layers in_data for the function, (check params).
        self.layers.insert(self.layer_count - 2, temp);//puts it back

        for i in (1..=self.layers.len() - 2).rev() { //loops through all layers backwards except last and first
            //how do i even explain.. check the comments in the forward function and it will make more sense.
            let temp = self.layers.remove(i + 1);
            let temp_two = self.layers.remove(i - 1);
            self.layers[i - 1].backward_costs(temp_two.get_out_data(), temp.get_costs());
            self.layers.insert(i - 1, temp_two);
            self.layers.insert(i + 1, temp);
        }

        //once batch size it reached then we add the gradients to teh weights.
        self.training_iterations += 1;
        if self.training_iterations % self.batch_size == 0 {
            //apply gradients
            for i in 0..self.layers.len() {
                if self.layers[i].is_trainable() {
                    //skip the first layer (always an input layer)
                    let pag = self.layers[i].get_params_and_grads();
                    let mut x = 0;
                    for j in pag.0 {
                        *j += (pag.1[x] / self.batch_size as f32) * self.learning_rate;
                        pag.1[x] = 0.0;
                        x += 1;
                    }
                }
            }
        }
    }
}
