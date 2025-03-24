pub trait GenericLayer {
    fn forward_data(&mut self, data: &Vec<f32>);
    fn backward_data(&mut self, data_in: &Vec<f32>, expected: &Vec<f32>);
    fn backward_costs(&mut self, data_in: &Vec<f32>, costs: &Vec<f32>);

    fn get_out_data(&self) -> &Vec<f32>;
    fn get_costs(&self) -> &Vec<f32>;
    fn get_weights(&self) -> &Vec<f32>;
    fn get_grads(&self) -> &Vec<f32>;
    fn get_in_size(&self) -> usize;
    fn get_out_size(&self) -> usize;
    fn get_weights_mut(&mut self) -> &mut Vec<f32>;
    fn get_params_and_grads(&mut self) -> (&mut Vec<f32>, &mut Vec<f32>);
    fn get_name(&self) -> String;
    fn is_trainable(&self) -> bool;
}
