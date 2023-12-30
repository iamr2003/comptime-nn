//goal, []type -> {type1, type2, ...}
const std = @import("std");

//this is some deep language garbage
//works for up to a fixed number of types
pub fn typeFlatten(comptime in_type: anytype) type {
    //from a blog post https://mht.wtf/post/comptime-struct/
    var fields: [in_type.len]std.builtin.Type.StructField = undefined;
    var union_fields: [in_type.len]std.builtin.Type.UnionField = undefined;

    // hacky painful answer for rn
    //I may make a script to generate zig, if I don't find a better way that this
    var uniqnames: [4][]const u8 = .{ "l1", "l2", "l3", "l4" };

    for (in_type, 0..) |t, i| {
        const default = t{};
        //I really wonder if we can make indexing work for us from here
        fields[i] = .{
            .name = uniqnames[i],
            .type = t,
            .default_value = &default,
            .is_comptime = false,
            .alignment = 0,
        };

        union_fields[i] = .{
            .name = uniqnames[i],
            .type = t,
            .alignment = 0,
        };
    }

    //this specific part is taken from blog post
    const flattened = @Type(.{ .Struct = .{
        .layout = .Auto,
        .fields = fields[0..],
        .decls = &[_]std.builtin.Type.Declaration{},
        .is_tuple = false,
    } });

    const together = @Type(.{ .Union = .{
        .layout = .Auto,
        .tag_type = null,
        .fields = union_fields[0..],
        .decls = &[_]std.builtin.Type.Declaration{},
    } });

    const indexer = struct {
        data: flattened = flattened{},
        //yep we'll hide more hackiness here
        pub fn index(data: flattened, i: u64) together {
            switch (i) {
                0 => data.l1,
                1 => data.l2,
                // 2 => data.l3,
                // 3 => data.l4,
                else => @compileError("too big for struct"),
            }
        }
    };
    return indexer;
}

// test "prim_tests" {
//     // try std.testing.expectEqual(@TypeOf(.{0,true,5}), actual: @TypeOf(expected))
//     const a = typeFlatten(.{ f64, bool });
//     const b = typeFlatten(.{ f64, bool });
//     try std.testing.expectEqual(a, b);
// }
