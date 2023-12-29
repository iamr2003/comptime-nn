const std = @import("std");

const g = @import("grad.zig");
const add = g.add;
const mul = g.mul;

//I might not need to be quite this silly
//might just implement the autograd way and use zig comptime to make it happen
//let's propogate through a single layer first
fn Layer(comptime input_nodes: usize, comptime output_nodes: usize, comptime in_activation: fn (in: g.GradVal) g.GradVal) type {
    const layerType = struct {
        weights: [output_nodes][input_nodes]f64 = [_][input_nodes]f64{[_]f64{1} ** input_nodes} ** output_nodes,
        comptime input_size: usize = input_nodes,
        comptime output_size: usize = output_nodes,

        pub fn forward(weights: [output_nodes][input_nodes]f64, input: [input_nodes]f64) [output_nodes]f64 {
            //will need to be converted into a gradval safe situation
            var output: [output_nodes]f64 = [_]f64{0} ** output_nodes;
            for (0..output_nodes) |output_index| {
                var sum: f64 = 0;
                for (input, 0..) |input_val, input_index| {
                    sum += weights[output_index][input_index] * input_val;
                }
                sum /= input_nodes;
                //yes inefficiency, will recompute with gradients
                output[output_index] = in_activation(g.GradVal{ .val = sum, .grad = 0 }).val;
            }
            return output;
        }

        //base_name needs to be used to ensure that each layer has uniquely named weights
        pub inline fn backward(weights: [output_nodes][input_nodes]f64, input: [input_nodes]g.GradVal, respect: u64, base_name: u64) [output_nodes]g.GradVal {
            var output: [output_nodes]g.GradVal = [_]g.GradVal{} ** output_nodes;
            for (0..output_nodes) |output_index| {
                var sum: g.GradVal = g.literal(0);
                inline for (input, 0..) |input_val, input_index| {
                    //if it was a id list, here is a uniqid
                    var id = base_name + output_index * input_nodes + input_index;
                    sum = add(sum, mul(g.variable_uuid(weights[output_index][input_index], id, respect), input_val));
                }
                sum = g.div(sum, input_nodes);
                //yes inefficiency, will recompute with gradients
                output[output_index] = in_activation(sum);
            }
            return output;
            //label each weight with a number
        }
    };
    return layerType;
}

//overall goal is to get gradient of each weight with respect to the evaluation metric
//gradient of eval metric for a weight is calculated based on the gradient of previous

inline fn relu(in: f64) f64 {
    return @max(0, in);
}

pub fn main() anyerror!void {
    // _ = nn;
    std.log.info("All your codebase are belong to us.", .{});
}

test "layer test single var" {
    const l1 = Layer(1, 1, g.relu);
    var layer: l1 = l1{};

    try std.testing.expectEqual(l1.forward(layer.weights, .{1}), .{1});
    try std.testing.expectEqual(l1.forward(layer.weights, .{5}), .{5});
    try std.testing.expectEqual(l1.forward(layer.weights, .{-1}), .{0});
}

test "single layer gradient test" {
    const l1 = Layer(2, 1, g.relu);
    var layer: l1 = l1{};

    try std.testing.expectEqual(l1.forward(layer.weights, .{ 1, 1 }), .{1});
    try std.testing.expectEqual(l1.forward(layer.weights, .{ 1, -1 }), .{0});
    try std.testing.expectEqual(l1.forward(layer.weights, .{ 2, -1 }), .{0.5});
    try std.testing.expectEqual(l1.forward(layer.weights, .{ 1, -2 }), .{0});

    try std.testing.expectEqual(l1.backward(layer.weights, .{ g.literal(1), g.literal(1) }, 0, 0)[0].grad, .{0.5});
    try std.testing.expectEqual(l1.backward(layer.weights, .{ g.literal(1), g.literal(1) }, 1, 0)[0].grad, .{0.5});
}
