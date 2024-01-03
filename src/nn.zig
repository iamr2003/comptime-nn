const std = @import("std");
const assert = std.debug.assert;

const g = @import("grad.zig");
const add = g.add;
const mul = g.mul;
const literal = g.literal;
const type_utils = @import("type_utils.zig");

pub fn Layer(
    comptime input_nodes: usize,
    comptime output_nodes: usize,
    comptime in_activation: *const fn (in: g.GradVal) g.GradVal,
) type {
    const layerType = struct {
        weights: [output_nodes][input_nodes]f64 = [_][input_nodes]f64{[_]f64{1} ** input_nodes} ** output_nodes,
        comptime param_count: usize = input_nodes * output_nodes,
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
                output[output_index] = in_activation(g.GradVal{ .val = sum, .grad = 0 }).val;
            }
            return output;
        }

        //base_name used to ensure that each layer has uniquely named weights
        pub inline fn backward(weights: [output_nodes][input_nodes]f64, input: [input_nodes]g.GradVal, respect: u64, base_name: u64) [output_nodes]g.GradVal {
            var output: [output_nodes]g.GradVal = [_]g.GradVal{.{ .val = 0, .grad = 0 }} ** output_nodes;
            for (0..output_nodes) |output_index| {
                var sum: g.GradVal = g.literal(0);
                inline for (input, 0..) |input_val, input_index| {
                    //if it was a id list, here is a uniqid
                    var id = base_name + output_index * input_nodes + input_index;
                    sum = add(sum, mul(g.variable_uuid(weights[output_index][input_index], id, respect), input_val));
                }
                sum = g.div(sum, g.literal(input_nodes));

                //lack of inlining here is not as terrible as it could be, as zero terms are elsewhere in tree
                output[output_index] = in_activation(sum);
            }
            return output;
        }
    };
    return layerType;
}

//given a list of compiler defined layers:
//have a forward function that chains all of them
//have a backward function that does same with weights
//include gradient step based on an eval function -- could be in a dependent trainer class
pub fn NN(comptime layer_types: anytype, comptime inputs: usize, comptime outputs: usize) type {
    //type inference and instantiation is fun
    //need to turn []type -> { type1, type2.., etc. }

    //replace with typeof at some point, and see if there are issues with defaults
    const layers_flat_types = type_utils.typeFlatten(layer_types);
    comptime var total_parameters: u64 = 0;
    comptime {
        for (layer_types) |t| {
            var init = t{};
            total_parameters += init.param_count;
        }
    }

    const nn_type = struct {
        layers: layers_flat_types = layers_flat_types{},
        comptime num_layers: usize = layer_types.len,

        pub fn forward(layers: layers_flat_types, input: [inputs]f64) [outputs]f64 {
            //we're just going to hardcode for now unfortunately
            //my shame -- compiler will ignore other cases at least
            switch (layer_types.len) {
                1 => {
                    return layer_types[0].forward(layers.l1.weights, input);
                },
                2 => {
                    return layer_types[1].forward(layers.l2.weights, layer_types[0].forward(layers.l1.weights, input));
                },
                3 => {
                    return layer_types[2].forward(layers.l3.weights, layer_types[1].forward(layers.l2.weights, layer_types[0].forward(layers.l1.weights, input)));
                },
                4 => {
                    return layer_types[3].forward(layers.l4.weights, layer_types[2].forward(layers.l3.weights, layer_types[1].forward(layers.l2.weights, layer_types[0].forward(layers.l1.weights, input))));
                },
                else => @compileError("not a supported number of nn layers"),
            }
        }

        pub fn backward(layers: layers_flat_types, input: [inputs]g.GradVal, respect: u64) [outputs]g.GradVal {
            switch (layer_types.len) {
                1 => {
                    return layer_types[0].backward(layers.l1.weights, input, respect, 0);
                },
                2 => {
                    return layer_types[1].backward(
                        layers.l2.weights,
                        layer_types[0].backward(layers.l1.weights, input, respect, 0),
                        respect,
                        layers.l1.param_count,
                    );
                },
                3 => {
                    return layer_types[2].backward(
                        layers.l3.weights,
                        layer_types[1].backward(
                            layers.l2.weights,
                            layer_types[0].backward(layers.l1.weights, input, respect, 0),
                            respect,
                            layers.l1.param_count,
                        ),
                        respect,
                        layers.l1.param_count + layers.l2.param_count,
                    );
                },
                4 => {
                    return layer_types[3].backward(
                        layers.l4.weights,
                        layer_types[2].backward(
                            layers.l3.weights,
                            layer_types[1].backward(
                                layers.l2.weights,
                                layer_types[0].backward(layers.l1.weights, input, respect, 0),
                                respect,
                                layers.l1.param_count,
                            ),
                            respect,
                            layers.l1.param_count + layers.l2.param_count,
                        ),
                        respect,
                        layers.l1.param_count + layers.l2.param_count + layers.l3.param_count,
                    );
                },
                else => @compileError("not a supported number of nn layers"),
            }
        }

        //updates the layers in place, output current loss
        pub fn train_step(
            layers: *layers_flat_types,
            batch_inputs: [][inputs]f64,
            batch_outputs: [][outputs]f64,
            loss_fn: *const fn ([outputs]g.GradVal, [outputs]g.GradVal) g.GradVal,
            scale: f64,
        ) f64 {
            std.debug.assert(batch_inputs.len == batch_outputs.len);
            std.debug.assert(batch_inputs.len > 0);

            //return new version of layers, aka weights
            //gradient of each weight with respect to the output
            var dweight_deval: [total_parameters]f64 = [_]f64{0} ** total_parameters;

            //take averaged gradient with respect to whole batch
            for (batch_inputs, 0..) |input, batch_id| {
                for (0..total_parameters) |weight_id| {
                    dweight_deval[weight_id] +=
                        loss_fn(g.vecToGrad(outputs, batch_outputs[batch_id]), backward(layers.*, g.vecToGrad(inputs, input), weight_id)).grad;
                }
            }

            var avg_loss: f64 = 0;
            for (batch_inputs, 0..) |input, batch_id| {
                avg_loss +=
                    loss_fn(g.vecToGrad(outputs, batch_outputs[batch_id]), backward(layers.*, g.vecToGrad(inputs, input), 0)).val;
            }

            avg_loss /= @floatFromInt(batch_inputs.len);

            //normalize for batch_size
            for (0..total_parameters) |i| {
                dweight_deval[i] /= @floatFromInt(batch_inputs.len);
            }

            //normalize overall gradient vector
            // var grad_mag: f64 = 0;
            // for (dweight_deval) |grad| {
            //     grad_mag += grad;
            // }
            //
            // for (0..total_parameters) |i| {
            //     dweight_deval[i] /= grad_mag;
            // }

            //do one step of gradient update

            //more hardcoding, that again compiles out-- otherwise indexing gets fun
            if (layer_types.len >= 1) {
                for (0..layers.l1.param_count) |weight_id| {
                    var i = weight_id;
                    var input_index = i % layers.l1.input_size;
                    var output_index = @divFloor(i, layers.l1.input_size);
                    layers.l1.weights[output_index][input_index] -= scale * dweight_deval[weight_id];
                }
            }

            if (layer_types.len >= 2) {
                for (0..layers.l2.param_count) |weight_id| {
                    var i = weight_id;
                    var offset = layers.l1.param_count;
                    var input_index = i % layers.l2.input_size;
                    var output_index = @divFloor(i, layers.l2.input_size);
                    layers.l2.weights[output_index][input_index] -= scale * dweight_deval[weight_id + offset];
                }
            }

            if (layer_types.len >= 3) {
                for (0..layers.l3.param_count) |weight_id| {
                    var i = weight_id;
                    var offset = layers.l1.param_count + layers.l2.param_count;
                    var input_index = i % layers.l3.input_size;
                    var output_index = @divFloor(i, layers.l3.input_size);
                    layers.l3.weights[output_index][input_index] -= scale * dweight_deval[weight_id + offset];
                }
            }
            if (layer_types.len >= 4) {
                for (0..layers.l4.param_count) |weight_id| {
                    var i = weight_id;
                    var offset = layers.l1.param_count + layers.l2.param_count + layers.l3.param_count;
                    var input_index = i % layers.l4.input_size;
                    var output_index = @divFloor(i, layers.l4.input_size);
                    layers.l4.weights[output_index][input_index] -= scale * dweight_deval[weight_id + offset];
                }
            }

            return avg_loss;
        }
    };

    return nn_type;
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

    try std.testing.expectEqual(l1.backward(layer.weights, .{ g.literal(1), g.literal(1) }, 0, 0)[0].grad, 0.5);
    try std.testing.expectEqual(l1.backward(layer.weights, .{ g.literal(1), g.literal(1) }, 1, 0)[0].grad, 0.5);
}

test "network 2 layers forward" {
    //might allow a different input format
    const nn1 = NN(.{ Layer(1, 1, g.relu), Layer(1, 1, g.relu) }, 1, 1);

    var nn: nn1 = nn1{};
    std.debug.print("\nType of nn:\n {}\n", .{@TypeOf(nn)});

    try std.testing.expectEqual(nn1.forward(nn.layers, .{1}), .{1});
    try std.testing.expectEqual(nn1.forward(nn.layers, .{5}), .{5});
    try std.testing.expectEqual(nn1.forward(nn.layers, .{-1}), .{0});
}

test "network complex layers" {
    const nn4 = NN(.{ Layer(3, 2, g.relu), Layer(2, 3, g.relu), Layer(3, 2, g.relu) }, 3, 2);
    var nn: nn4 = nn4{};

    try std.testing.expectEqual(nn4.forward(nn.layers, .{ 1, 2, -3 }), .{ 0, 0 });
    try std.testing.expectEqual(nn4.forward(nn.layers, .{ 1, 2, 3 }), .{ 2, 2 });
}

test "network multilayer gradient test" {
    const nn3 = NN(.{ Layer(2, 2, g.relu), Layer(2, 2, g.relu) }, 2, 2);
    var nn: nn3 = nn3{};

    //note numbers change with not 1/1 as inputs, as they should
    // this is not the most nuanced test, but I don't feel like hand calculating more derivatives
    try std.testing.expectEqual(nn3.backward(nn.layers, .{ literal(1), literal(1) }, 4)[0].grad, 0.5);
    try std.testing.expectEqual(nn3.backward(nn.layers, .{ literal(1), literal(1) }, 0)[0].grad, 0.25);
    try std.testing.expectEqual(nn3.backward(nn.layers, .{ literal(1), literal(1) }, 3)[0].grad, 0.25);
    try std.testing.expectEqual(nn3.backward(nn.layers, .{ literal(1), literal(1) }, 7)[0].grad, 0);
    try std.testing.expectEqual(nn3.backward(nn.layers, .{ literal(1), literal(1) }, 7)[1].grad, 0.5);
}

fn lin_func(comptime slope: f64, in: [1]f64) [1]f64 {
    return .{slope * in[0]};
}

fn square_loss(expect: [1]g.GradVal, actual: [1]g.GradVal) g.GradVal {
    var diff = g.sub(expect[0], actual[0]);
    return g.mul(diff, diff);
}

//there are some issues, so let's try to create minimal reproducible small gradient steps
test "no loss test" {
    //the most basic, 1 weight, learning a linear relationship
    const nn_t = NN(.{Layer(1, 1, g.linear)}, 1, 1);
    var nn: nn_t = nn_t{};

    var correct_in: [4][1]f64 = .{ .{5}, .{2.3}, .{-10}, .{-800.62} };
    var correct_out: [4][1]f64 = .{ .{5}, .{2.3}, .{-10}, .{-800.62} };

    var correct_loss = nn_t.train_step(&nn.layers, &correct_in, &correct_out, square_loss, 0.01);
    //verify no loss, and no changed vars
    try std.testing.expectEqual(correct_loss, 0);
    try std.testing.expectEqual(nn.layers.l1.weights[0][0], 1);
}

test "no loss test multilayer" {
    //the most basic, 1 weight, learning a linear relationship
    const nn_t = NN(.{ Layer(1, 1, g.linear), Layer(1, 1, g.linear), Layer(1, 1, g.linear), Layer(1, 1, g.linear) }, 1, 1);
    var nn: nn_t = nn_t{};

    var correct_in: [4][1]f64 = .{ .{5}, .{2.3}, .{-10}, .{-800.62} };
    var correct_out: [4][1]f64 = .{ .{5}, .{2.3}, .{-10}, .{-800.62} };

    var correct_loss = nn_t.train_step(&nn.layers, &correct_in, &correct_out, square_loss, 0.01);
    //verify no loss, and no changed vars
    try std.testing.expectEqual(correct_loss, 0);
    try std.testing.expectEqual(nn.layers.l1.weights[0][0], 1);
    try std.testing.expectEqual(nn.layers.l2.weights[0][0], 1);
    try std.testing.expectEqual(nn.layers.l3.weights[0][0], 1);
    try std.testing.expectEqual(nn.layers.l4.weights[0][0], 1);
}

test "simple linear single layer training" {
    //the most basic, 1 weight, learning a linear relationship
    const nn_t = NN(.{Layer(1, 1, g.linear)}, 1, 1);
    var nn: nn_t = nn_t{};

    var correct_in: [1][1]f64 = .{.{5}};
    var correct_out: [1][1]f64 = .{.{10}};

    var loss_1 = nn_t.train_step(&nn.layers, &correct_in, &correct_out, square_loss, 0.1);

    // move in correct direction
    try std.testing.expectEqual(loss_1, 25); //5*5
    try std.testing.expectEqual(nn.layers.l1.weights[0][0], 6); // 5 * 2(5-10) * -1 * 0.1

    var loss_2 = nn_t.train_step(&nn.layers, &correct_in, &correct_out, square_loss, 0.1);

    try std.testing.expectEqual(loss_2, 400); //((6*5)-10)**2
    try std.testing.expectEqual(nn.layers.l1.weights[0][0], -14); // 6 * 2(30-10) * -1 * 0.1
    //correct directions, would eventually converge
}

test "simple linear multilayer training" {
    //2 weights, in sequence
    const nn_t = NN(.{ Layer(1, 1, g.linear), Layer(1, 1, g.linear) }, 1, 1);
    var nn: nn_t = nn_t{};

    var correct_in: [1][1]f64 = .{.{5}};
    var correct_out: [1][1]f64 = .{.{10}};

    var loss_1 = nn_t.train_step(&nn.layers, &correct_in, &correct_out, square_loss, 0.1);

    // move in correct direction
    try std.testing.expectEqual(loss_1, 25); //5*5

    //weights are basically same situation here
    //because no nonlinearity, weights end up doing exact same thing
    try std.testing.expectEqual(nn.layers.l2.weights[0][0], 6); // 5 * 2(5-10) * -1 * 0.1
    try std.testing.expectEqual(nn.layers.l1.weights[0][0], 6); // 5 * 2(5-10) * -1 * 0.1

    var loss_2 = nn_t.train_step(&nn.layers, &correct_in, &correct_out, square_loss, 0.01);

    try std.testing.expectEqual(loss_2, 28900); //((6*5*6)-10)**2

    //check correct direction
    try std.testing.expect(nn.layers.l1.weights[0][0] < 6);
    try std.testing.expect(nn.layers.l2.weights[0][0] < 6);
}

test "simple non linear single layer training positive" {
    //the most basic, 1 weight, with a relu relationship
    const nn_t = NN(.{Layer(1, 1, g.linear)}, 1, 1);
    var nn: nn_t = nn_t{};

    //with positive should do exactly the same thing

    var correct_in: [1][1]f64 = .{.{5}};
    var correct_out: [1][1]f64 = .{.{10}};

    var loss_1 = nn_t.train_step(&nn.layers, &correct_in, &correct_out, square_loss, 0.1);

    // move in correct direction
    try std.testing.expectEqual(loss_1, 25); //5*5
    try std.testing.expectEqual(nn.layers.l1.weights[0][0], 6); // 5 * 2(5-10) * -1 * 0.1

    var loss_2 = nn_t.train_step(&nn.layers, &correct_in, &correct_out, square_loss, 0.1);

    try std.testing.expectEqual(loss_2, 400); //((6*5)-10)**2
    try std.testing.expectEqual(nn.layers.l1.weights[0][0], -14); // 6 * 2(30-10) * -1 * 0.1
}

//do a test with relu, then linear on a negative val

test "simple noninear, linear" {
    const nn_t = NN(.{Layer(1, 1, g.relu),Layer(1, 1, g.linear)}, 1, 1);
    var nn: nn_t = nn_t{};

    //with positive should do exactly the same thing

    var correct_in: [1][1]f64 = .{.{5}};
    var correct_out: [1][1]f64 = .{.{-10}};

    var loss_1 = nn_t.train_step(&nn.layers, &correct_in, &correct_out, square_loss, 0.1);

    // move in correct direction
    try std.testing.expectEqual(loss_1, 225); //5*5
    try std.testing.expectEqual(nn.layers.l1.weights[0][0], -14); // 5 * 2(5+10) * -1 * 0.1
    try std.testing.expectEqual(nn.layers.l2.weights[0][0], -14); // 5 * 2(5+10) * -1 * 0.1

    var loss_2 = nn_t.train_step(&nn.layers, &correct_in, &correct_out, square_loss, 0.1);

    try std.testing.expectEqual(loss_2, 100); //((0)-10)**2
    // try std.testing.expectEqual(nn.layers.l1.weights[0][0], -14); // 6 * 2(30-10) * -1 * 0.

    try std.testing.expectEqual(nn.layers.l2.weights[0][0], -14); // had no effect
    try std.testing.expectEqual(nn.layers.l1.weights[0][0], -14); // had no effect
}

//next tests should be with multinode, multi layer
//bc this is where I suspect there may be an issue

//the most basic, 1 weight, with a relu relationship
