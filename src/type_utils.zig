//goal, []type -> {type1, type2, ...}
const std = @import("std");

//this is some deep language garbage
pub fn typeFlatten(comptime in_type: anytype) type {
    //from a blog post https://mht.wtf/post/comptime-struct/
    var fields: [in_type.len]std.builtin.Type.StructField = undefined;

    // hacky answer for rn
    var uniqnames: [4][]const u8 = .{ "l1", "l2", "l3", "l4" };

    for (in_type, 0..) |t, i| {
        fields[i] = .{
            .name = uniqnames[i],
            .type = t,
            .default_value = t{},
            .is_comptime = false,
            .alignment = 0,
        };
    }
    return @Type(.{ .Struct = .{
        .layout = .Auto,
        .fields = fields[0..],
        .decls = &[_]std.builtin.Type.Declaration{},
        .is_tuple = false,
    } });
}

// test "prim_tests" {
//     // try std.testing.expectEqual(@TypeOf(.{0,true,5}), actual: @TypeOf(expected))
//     const a = typeFlatten(.{ f64, bool });
//     const b = typeFlatten(.{ f64, bool });
//     try std.testing.expectEqual(a, b);
// }
