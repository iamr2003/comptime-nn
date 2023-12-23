const std = @import("std");

//I might not need to be quite this silly
//might just implement the autograd way and use zig comptime to make it happen
fn Layer(comptime input_nodes: usize, comptime output_nodes: usize, comptime in_activation: fn (in: f64) f64) type {
    const layerType = struct {
        weights: [output_nodes][input_nodes]f64 = [_][input_nodes]f64{[_]f64{1} ** input_nodes} ** output_nodes,
        comptime input_size: usize = input_nodes,
        comptime output_size: usize = output_nodes,

        pub fn forward(weights: [output_nodes][input_nodes]f64, input: [input_nodes]f64) [output_nodes]f64 {
            var output: [output_nodes]f64 = [_]f64{0} ** output_nodes;
            for (0..output_nodes) |output_index| {
                var sum: f64 = 0;
                for (input, 0..) |input_val, input_index| {
                    sum += weights[output_index][input_index] * input_val;
                }
                sum /= input_nodes;
                output[output_index] = in_activation(sum);
            }
            return output;
        }
    };
    return layerType;
}

fn Network(comptime input_layers: type) type {
    comptime {
        //this is so dumb and hacky
        const la: input_layers = input_layers{};
        //need to fix this to be an array
        const network_type = struct {
            comptime layer: input_layers = input_layers{},
            // hardcode dumbness, also annoyances of functions being from type
            // and not member
            pub fn forward(l: input_layers, in: [la.input_size]f64) [la.output_size]f64 {
                return input_layers.forward(l.weights, in);
            }
        };
        return network_type;
    }
}

fn relu(in: f64) f64 {
    return @max(0, in);
}

pub fn main() anyerror!void {
    // _ = nn;
    std.log.info("All your codebase are belong to us.", .{});
}

test "layer test single var" {
    const l1 = Layer(1, 1, relu);
    var layer: l1 = l1{};

    // var nn: nn_type = .{ .layers = @TypeOf(nn_type.layers).init() };

    try std.testing.expectEqual(l1.forward(layer.weights, .{1}), .{1});
    try std.testing.expectEqual(l1.forward(layer.weights, .{5}), .{5});
    try std.testing.expectEqual(l1.forward(layer.weights, .{-1}), .{0});
}

test "nn test single var single layer" {
    const nn_type = Network(Layer(1, 1, relu));
    var nn: nn_type = nn_type{};
    // //function not preserved?
    try std.testing.expectEqual(nn_type.forward(nn.layer, .{-1}), .{0});
    try std.testing.expectEqual(nn_type.forward(nn.layer, .{1}), .{1});
}
