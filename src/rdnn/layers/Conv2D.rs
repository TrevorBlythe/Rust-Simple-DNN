use super::GenericLayerTrait::GenericLayer;
use rand::Rng;

pub struct Conv2D {
    pub filters_weights: Vec<f32>, //filter weights, parameters.. whatever you wanna call it
    pub filter_grads: Vec<f32>,
    pub biases: Vec<f32>,
    pub bias_grads: Vec<f32>,
    pub output_data: Vec<f32>,
    pub input_grads: Vec<f32>,
    pub input_channels: usize, 
    pub output_channels: usize, //also represents filter count
    pub filter_width: usize,
    pub filter_height: usize,
    pub input_height: usize,
    pub input_width: usize,
    pub output_height: usize,
    pub output_width: usize,
    pub stride: usize,

    // Precomputed constants for optimization
    output_area: usize,
    input_area: usize,
    filter_area: usize,
    filter_area_times_channels: usize,
}

impl GenericLayer for Conv2D {
    fn get_name(&self) -> &str {
        "Conv2D"
    }

    fn get_input_grads(&self) -> &Vec<f32> {
        &self.input_grads
    }

    fn get_out_data(&self) -> &Vec<f32> {
        &self.output_data
    }

    fn is_trainable(&self) -> bool {
        true
    }

    fn backward_target(&mut self, input_data: &Vec<f32>, expected: &Vec<f32>) {
        //sorry for the layer of indirection here, i am NOT typing that monstrosity twice (the real backwards function is below)
        self.backward(input_data, &expected.iter().zip(self.output_data.iter()).map(|(e, o)| e - o).collect::<Vec<f32>>());
    }

    fn backward_grads(&mut self, input_data: &Vec<f32>, grads: &Vec<f32>) {
        //sorry for the layer of indirection here, i am NOT typing that monstrosity twice (the real backwards function is below)
        self.backward(input_data, &grads);
    }

    fn forward_data(&mut self, input_data: &Vec<f32>) {
        self.forward(input_data);
    }

    fn get_params_and_grads(&mut self) -> Vec<(&mut Vec<f32>, &mut Vec<f32>)> {
        vec![(&mut self.filters_weights, &mut self.filter_grads), (&mut self.biases, &mut self.bias_grads)]
    }

    fn get_params_mut(&mut self) -> Vec<&mut Vec<f32>> {
        vec![&mut self.filters_weights, &mut self.biases]
    }

    fn get_grads(&self) -> Vec<&Vec<f32>> {
        vec![&self.filter_grads, &self.bias_grads]
    }

    fn get_params(&self) -> Vec<&Vec<f32>> {
        vec![&self.filters_weights, &self.biases]
    }

    fn get_in_size(&self) -> usize {
        self.input_channels * self.input_height * self.input_width
    }

    fn get_out_size(&self) -> usize {
        self.output_channels * self.output_height * self.output_width
    }
}


pub fn new(
        input_dims: [usize; 3],
        filter_dims: [usize; 2],
        output_channels: usize, //also represents the amount of filters
        stride: usize,
    ) -> Box<Conv2D> {
        let input_width = input_dims[0];
        let input_height = input_dims[1];
        let input_channels = input_dims[2];
        let filter_width = filter_dims[0];
        let filter_height = filter_dims[1];

        //since output height and width it gets floored 
        let output_height:usize = ((input_height - filter_height + 1) as f32 / stride as f32).ceil() as usize;
        let output_width:usize = ((input_width - filter_width + 1) as f32 / stride as f32).ceil() as usize;
        let filters_size = output_channels * input_channels * filter_width * filter_height;

        // if (input_height as f32 - (filter_height as f32) + 1.0) / stride as f32
        // != output_height as f32 {
        //     println!("Possible mismatch of input size,
        //     This will cause some areas of the input to be ignored in convolution\n
        //     output_height: {:?}, output_width: {:?}", output_height, output_width)
        // }

        let ret = Box::new(Conv2D {
            filters_weights: (0..filters_size).map(|_| rand::rng().random_range(-1.0..1.0) / (input_channels as f32)).collect(),
            filter_grads: vec![0.0; filters_size],
            biases: (0..output_height * output_width * output_channels).map(|_| 0.1 * rand::rng().random_range(-1.0..1.0)).collect(),
            bias_grads: vec![0.0; output_height * output_width * output_channels],
            output_data: vec![0.0; output_channels * output_height * output_width],
            input_grads: vec![0.0; input_channels * input_height * input_width],
            input_channels,
            output_channels,
            filter_width,
            filter_height,
            input_height,
            input_width,
            output_height,
            output_width,
            stride,
            output_area: output_height * output_width,
            input_area: input_height * input_width,
            filter_area: filter_width * filter_height,
            filter_area_times_channels: filter_width * filter_height * input_channels,
        });

        return ret;
    }

impl Conv2D {

    fn forward(&mut self, input_data: &Vec<f32>) {
        self.output_data.fill(0.0);
        for filterIndex in 0..self.output_channels{
            let outputAreaOffset = filterIndex * self.output_area;
            let filterVolumeOffset = filterIndex * self.filter_area_times_channels;
            for outputHeightIndex in 0..self.output_height{
                let filterVerticalIndexOnInput = outputHeightIndex * self.stride;
                let filterVerticalOffsetOnInput = outputHeightIndex * self.output_width;
                for outputWidthIndex in 0..self.output_width{
                    let outputIndex = outputWidthIndex + filterVerticalOffsetOnInput + outputAreaOffset;
                    let filterHorizontalIndexOnInput = outputWidthIndex * self.stride;
                    self.output_data[outputIndex] += self.biases[outputIndex];
                    for channelIndex in 0..self.input_channels{
                        let filterInputChannelsOffsetOnInput = channelIndex * self.input_area + filterHorizontalIndexOnInput;
                        let filterChannelsOffset = channelIndex * self.filter_area + filterVolumeOffset;

                            for filterHeightIndex in 0..self.filter_height{
                                let specificfilterSquareVerticalIndexOnInput = 
                                (filterHeightIndex + filterVerticalIndexOnInput) * self.input_width + filterInputChannelsOffsetOnInput;
                                let specificfilterSquareAreaAndVerticalOffset = 
                                filterHeightIndex * self.filter_width + filterChannelsOffset;

                                for filterWidthIndex in 0..self.filter_width{
                                    self.output_data[outputIndex] += 
                                    input_data[filterWidthIndex + specificfilterSquareVerticalIndexOnInput] *
                                    self.filters_weights[filterWidthIndex + specificfilterSquareAreaAndVerticalOffset];
                                }
                            }
                    }
                }
            }
        }
    }

    fn backward(&mut self, input_data: &Vec<f32>, grads: &Vec<f32>) {
        self.input_grads.fill(0.0);

        //brain hurty...
        for filterIndex in 0..self.output_channels{
            let outputAreaOffset = filterIndex * self.output_area;
            let filterVolumeOffset = filterIndex * self.filter_area_times_channels;
            for outputHeightIndex in 0..self.output_height{
                let filterVerticalIndexOnInput = outputHeightIndex * self.stride;
                let filterVerticalOffsetOnInput = outputHeightIndex * self.output_width;
                for outputWidthIndex in 0..self.output_width{
                    let outputIndex = outputWidthIndex + filterVerticalOffsetOnInput + outputAreaOffset;
                    let filterHorizontalIndexOnInput = outputWidthIndex * self.stride;
                    self.bias_grads[outputIndex] += grads[outputIndex];
                    for channelIndex in 0..self.input_channels{
                        let filterInputChannelsOffsetOnInput = channelIndex * self.input_area + filterHorizontalIndexOnInput;
                        let filterChannelsOffset = channelIndex * self.filter_area + filterVolumeOffset;

                            for filterHeightIndex in 0..self.filter_height{
                                let specificfilterSquareVerticalIndexOnInput = 
                                (filterHeightIndex + filterVerticalIndexOnInput) * self.input_width + filterInputChannelsOffsetOnInput;
                                let specificfilterSquareAreaAndVerticalOffset = 
                                filterHeightIndex * self.filter_width + filterChannelsOffset;

                                for filterWidthIndex in 0..self.filter_width{
                                    self.filter_grads[filterWidthIndex + specificfilterSquareAreaAndVerticalOffset] += 
                                    input_data[filterWidthIndex + specificfilterSquareVerticalIndexOnInput] * grads[outputIndex];
                                    self.input_grads[filterWidthIndex + specificfilterSquareVerticalIndexOnInput] += 
                                    self.filters_weights[filterWidthIndex + specificfilterSquareAreaAndVerticalOffset] *
                                    grads[outputIndex];
                                }
                            }
                    }
                }
            }
        }

       
    }
}