// pub const cosh = @import("complex/cosh.zig").sinh;
// pub const tanh = @import("complex/tanh.zig").tanh;
const std = @import("std");

//may all become SIMD'd at some point
//this is the traditional way to do it, basically same code as micrograd
pub const GradVal = struct {
    val: f64 = 0,
    grad: f64 = 0,
    // with_respect_to: ?*const []u8 = null,
    // could change this to have multiple grad values, which would be the way
};

//need to store NAME of var,
//still a lot of questions about multi dimensions with weights

//give basic binary ops, building up in the traditional way
pub inline fn add(lhs: GradVal, rhs: GradVal) GradVal {
    return GradVal{ .val = lhs.val + rhs.val, .grad = lhs.grad + rhs.grad };
}

pub inline fn sub(lhs: GradVal, rhs: GradVal) GradVal {
    return GradVal{ .val = lhs.val - rhs.val, .grad = lhs.grad - rhs.grad };
}

pub inline fn mul(lhs: GradVal, rhs: GradVal) GradVal {
    return GradVal{
        .val = lhs.val * rhs.val,
        .grad = lhs.grad * rhs.val + lhs.val * rhs.grad,
    };
}

pub inline fn div(lhs: GradVal, rhs: GradVal) GradVal {
    //nan issues
    return GradVal{
        .val = lhs.val / rhs.val,
        .grad = ((rhs.val * lhs.grad) - (lhs.val * rhs.grad)) / (rhs.val * rhs.val),
    };
}

//goal is a lot of inefficiency compiles out
//we are going to duplicate expression a million times, perhaps leaving tree intact would be more helpful
pub inline fn variable(val: f64, name: []const u8, respect: []const u8) GradVal {
    return GradVal{ .val = val, .grad = if (std.mem.eql(u8, name, respect)) 1 else 0 };
}

//version of variable that uses integers to label, since this is much easier to generate
pub inline fn variable_uuid(val: f64, name: u64, respect: u64) GradVal {
    return GradVal{ .val = val, .grad = if (name == respect) 1 else 0 };
}

pub inline fn literal(val: f64) GradVal {
    return GradVal{ .val = val, .grad = 0 };
}

pub inline fn max(lhs: GradVal, rhs: GradVal) GradVal {
    if (lhs.val < rhs.val) {
        return GradVal{ .val = rhs.val, .grad = rhs.grad };
    } else {
        return GradVal{ .val = lhs.val, .grad = lhs.grad };
    }
}

pub inline fn min(lhs: GradVal, rhs: GradVal) GradVal {
    if (lhs.val > rhs.val) {
        return GradVal{ .val = rhs.val, .grad = rhs.grad };
    } else {
        return GradVal{ .val = lhs.val, .grad = lhs.grad };
    }
}

//can write out arbitrary ones, but this is easier
pub inline fn exp(in: GradVal) GradVal {
    return GradVal{ .val = @exp(in.val), .grad = @exp(in.val) };
}

// inline fn tanhg(in:GradVal)GradVal{
//     return GradVal{
//         .val = tanh(in.val),
//         .grad = sech(in.val)
//     };
// }

//need exp for sigmoid
//tanh, ELU other common ones
//RETURN TO INLINE WHEN POSSIBLE
pub fn relu(in: GradVal) GradVal {
    return max(literal(0), in);
}

pub fn sigmoid(in: GradVal) GradVal {
    //need some safeties I think
    return div(exp(in), add(literal(1), exp(in)));
}

//utils

pub fn vecToGrad(comptime size: usize, in: [size]f64) [size]GradVal {
    var out: [size]GradVal = undefined;
    for (in, 0..) |v, i| {
        out[i] = literal(v);
    }
    return out;
}

//examples
fn square(val: f64) GradVal {
    return mul(variable(val, "x", "x"), variable(val, "x", "x"));
}

fn abs(val: f64) GradVal {
    return max(variable(val, "x", "x"), mul(literal(-1), variable(val, "x", "x")));
}

test "simple single var" {
    try std.testing.expectEqual(square(2).grad, (GradVal{ .val = 4, .grad = 4 }).grad);
    try std.testing.expectEqual(square(5), GradVal{ .val = 25, .grad = 10 });

    try std.testing.expectEqual(abs(7), GradVal{ .val = 7, .grad = 1 });
    try std.testing.expectEqual(abs(-7), GradVal{ .val = 7, .grad = -1 });
}

fn multivar(x: f64, y: f64, respect: []const u8) GradVal {
    //x^2y + y^2
    // dx is 2xy
    //dy is x^2 + 2y
    return add(mul(mul(variable(x, "x", respect), variable(x, "x", respect)), variable(y, "y", respect)), mul(variable(y, "y", respect), variable(y, "y", respect)));
}
test "simple multivar" {
    try std.testing.expectEqual(multivar(2, 3, "x"), GradVal{ .val = 21, .grad = 12 });
    try std.testing.expectEqual(multivar(2, 3, "y"), GradVal{ .val = 21, .grad = 10 });
}
