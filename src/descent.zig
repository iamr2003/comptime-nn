const std = @import("std");
//let's do some gradient descent to make sure this does what I think it does

//2d function with minimum at 0,4
// x^2 + (y-4)^2
const g = @import("grad.zig");
const add = g.add;
const mul = g.mul;
const sub = g.sub;
const variable = g.variable;

fn bowl(x: f64, y: f64, respect: []const u8) g.GradVal {
    return add(mul(variable(x, "x", respect), variable(x, "x", respect)), mul(sub(variable(y, "y", respect), g.literal(4)), sub(variable(y, "y", respect), g.literal(4))));
}

pub fn main() !void {
    //euclidean
    const step_size = 0.1;

    const steps = 100;
    var curr_x: f64 = 1;
    var curr_y: f64 = 1;

    std.debug.print("INIT: x: {} y:{} \n", .{ curr_x, curr_y });
    for (0..steps) |step| {
        var dx = bowl(curr_x, curr_y, "x").grad;
        var dy = bowl(curr_x, curr_y, "y").grad;
        var d_len = @sqrt(dx * dx + dy * dy);

        curr_x -= (dx / d_len) * step_size;
        curr_y -= (dy / d_len) * step_size;
        std.debug.print("x: {} y:{} step:{}\n", .{ curr_x, curr_y, step });
    }
}
