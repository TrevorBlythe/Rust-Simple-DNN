

/*
Represents a layer in a neural network
*/
pub trait GenericLayer {

//The CORE operations ------- (todo they should take slices (more flexible))
    fn forward_data(&mut self, data: &Vec<f32>); //Input data into the model, returns output

    //"data_in" represents the previous layers output. This gets passed in to avoid
    //having to make some linked list structure and to keep everything modular
    fn backward_target(&mut self, data_in: &Vec<f32>, expected: &Vec<f32>); 
    //"grads" here, represents the error of each output f32 in the vector of the layers output
    fn backward_grads(&mut self, data_in: &Vec<f32>, grads: &Vec<f32>);

//Getters and utils
    fn get_in_size(&self) -> usize;
    fn get_out_size(&self) -> usize;
    fn get_out_data(&self) -> &Vec<f32>; //Gets the output of the layer
    fn get_input_grads(&self) -> &Vec<f32>; //Loss of the layers input 
    fn get_name(&self) -> &str; //Name, just for debugging
    fn is_trainable(&self) -> bool; //Some layers like Sigmoid do not have weights
    //fn get_input() <-- DISCLAIMER: doesnt exist
    //Layers dont store inputs, its already stored in prev. layer outputs
    //or in the case of InputLayers, its output is always its input.


    //params and grads are one dimensional vecs, these functions
    //return vectors of these vecs cause some layers like FC,
    //have more then one set of params (weights AND biases)
    fn get_params(&self) -> Vec<&Vec<f32>>; //Any weights the layer might have (layers without weights will return empty)
    fn get_grads(&self) -> Vec<&Vec<f32>>; //Gradients for each parameter
    fn get_params_mut(&mut self) -> Vec<&mut Vec<f32>>; //A vector of param vectors
    //Returns a vector of param grad pairs, held in tuple (params,grads), this is used by the network struct to apply grads to weights
    fn get_params_and_grads(&mut self) -> Vec<(&mut Vec<f32>, &mut Vec<f32>)>; //param[i] corresponds to grad[i]

}