// =============================================================================
// rf_classifier.v
// =============================================================================
`timescale 1ns/1ps

module custom_comparator #(
    parameter integer DW = 16,
    parameter signed [DW-1:0] THRESHOLD = 16'sd0
) (
    input  wire signed [DW-1:0] x,
    output wire                 y
);
    
    generate
        if (THRESHOLD == -16'sd3840) begin : FAST_NEG_375
            assign y = (x[15:8] == 8'b1111_0000);
        end else if (THRESHOLD == -16'sd256) begin : FAST_NEG_025
            assign y = x[15] & ~(&x[15:8]);
        end else if (THRESHOLD == 16'sd256) begin : FAST_POS_025
            assign y = x[15] | (x[15:8] == 8'b0000_0000);
        end else begin : GENERIC
            assign y = ($signed(x) < THRESHOLD);
        end
    endgenerate
endmodule



module decision_tree #(
    parameter integer DW     = 16,
    parameter integer N_COEF = 14,
    parameter integer TREE_ID = 0
) (
    input  wire [N_COEF*DW-1:0] feat_flat,
    output wire                 vote // 1 = CHF, 0 = normal
);

    wire signed [DW-1:0] feat0  = feat_flat[1*DW-1  -: DW];
    wire signed [DW-1:0] feat1  = feat_flat[2*DW-1  -: DW];
    wire signed [DW-1:0] feat2  = feat_flat[3*DW-1  -: DW];
    wire signed [DW-1:0] feat3  = feat_flat[4*DW-1  -: DW];
    wire signed [DW-1:0] feat4  = feat_flat[5*DW-1  -: DW];
    wire signed [DW-1:0] feat5  = feat_flat[6*DW-1  -: DW];
    wire signed [DW-1:0] feat6  = feat_flat[7*DW-1  -: DW];
    wire signed [DW-1:0] feat7  = feat_flat[8*DW-1  -: DW];
    wire signed [DW-1:0] feat8  = feat_flat[9*DW-1  -: DW];
    wire signed [DW-1:0] feat9  = feat_flat[10*DW-1 -: DW];
    wire signed [DW-1:0] feat10 = feat_flat[11*DW-1 -: DW];
    wire signed [DW-1:0] feat11 = feat_flat[12*DW-1 -: DW];
    wire signed [DW-1:0] feat12 = feat_flat[13*DW-1 -: DW];
    wire signed [DW-1:0] feat13 = feat_flat[14*DW-1 -: DW];

  
    generate
        if (TREE_ID == 0) begin : TREE0
            wire n0; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd3706)) c0 (.x(feat11), .y(n0)); // node 0: feat11 < -3706 ?
            wire n1; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd2520)) c1 (.x(feat4), .y(n1)); // node 1: feat4 < -2520 ?
            wire n2; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd3948)) c2 (.x(feat1), .y(n2)); // node 3: feat1 < -3948 ?
            wire n3; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd4310)) c3 (.x(feat4), .y(n3)); // node 6: feat4 < -4310 ?
            wire n4; custom_comparator #(.DW(DW), .THRESHOLD(16'sd8600)) c4 (.x(feat6), .y(n4)); // node 8: feat6 < 8600 ?
            wire n5; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd4428)) c5 (.x(feat7), .y(n5)); // node 9: feat7 < -4428 ?
            wire n6; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd4626)) c6 (.x(feat13), .y(n6)); // node 11: feat13 < -4626 ?
            wire n7; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd4520)) c7 (.x(feat8), .y(n7)); // node 13: feat8 < -4520 ?
            assign vote = (n0 ? (n1 ? 1'b1 : (n2 ? 1'b1 : 1'b0)) : (n3 ? 1'b1 : (n4 ? (n5 ? 1'b1 : (n6 ? 1'b1 : (n7 ? 1'b1 : 1'b0))) : 1'b1)));
        end else if (TREE_ID == 1) begin : TREE1
            wire n0; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd3702)) c0 (.x(feat12), .y(n0)); // node 0: feat12 < -3702 ?
            wire n1; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd3558)) c1 (.x(feat1), .y(n1)); // node 1: feat1 < -3558 ?
            wire n2; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd3312)) c2 (.x(feat4), .y(n2)); // node 3: feat4 < -3312 ?
            wire n3; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd4118)) c3 (.x(feat9), .y(n3)); // node 6: feat9 < -4118 ?
            wire n4; custom_comparator #(.DW(DW), .THRESHOLD(16'sd8326)) c4 (.x(feat12), .y(n4)); // node 8: feat12 < 8326 ?
            wire n5; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd4562)) c5 (.x(feat4), .y(n5)); // node 9: feat4 < -4562 ?
            wire n6; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd4354)) c6 (.x(feat8), .y(n6)); // node 11: feat8 < -4354 ?
            wire n7; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd4446)) c7 (.x(feat7), .y(n7)); // node 13: feat7 < -4446 ?
            wire n8; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd5010)) c8 (.x(feat1), .y(n8)); // node 15: feat1 < -5010 ?
            assign vote = (n0 ? (n1 ? 1'b1 : (n2 ? 1'b1 : 1'b0)) : (n3 ? 1'b1 : (n4 ? (n5 ? 1'b1 : (n6 ? 1'b1 : (n7 ? 1'b1 : (n8 ? 1'b1 : 1'b0)))) : 1'b1)));
        end else if (TREE_ID == 2) begin : TREE2
            wire n0; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd3690)) c0 (.x(feat11), .y(n0)); // node 0: feat11 < -3690 ?
            wire n1; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd3070)) c1 (.x(feat2), .y(n1)); // node 1: feat2 < -3070 ?
            wire n2; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd4206)) c2 (.x(feat3), .y(n2)); // node 4: feat3 < -4206 ?
            wire n3; custom_comparator #(.DW(DW), .THRESHOLD(16'sd8514)) c3 (.x(feat3), .y(n3)); // node 6: feat3 < 8514 ?
            wire n4; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd4430)) c4 (.x(feat6), .y(n4)); // node 7: feat6 < -4430 ?
            wire n5; custom_comparator #(.DW(DW), .THRESHOLD(-16'sd4610)) c5 (.x(feat8), .y(n5)); // node 9: feat8 < -4610 ?
            wire n6; custom_comparator #(.DW(DW), .THRESHOLD(16'sd8568)) c6 (.x(feat8), .y(n6)); // node 11: feat8 < 8568 ?
            assign vote = (n0 ? (n1 ? 1'b1 : 1'b1) : (n2 ? 1'b1 : (n3 ? (n4 ? 1'b1 : (n5 ? 1'b1 : (n6 ? 1'b0 : 1'b1))) : 1'b1)));
        end
    endgenerate
endmodule


module rf_classifier #(
    parameter integer DW        = 16,
    parameter integer N_COEF    = 14,
    parameter integer NUM_TREES = 3
) (
    input  wire [N_COEF*DW-1:0] feat_flat,
    output wire                 chf_detected // majority vote across all trees
);
    wire [NUM_TREES-1:0] votes;

    genvar t;
    generate
        for (t = 0; t < NUM_TREES; t = t + 1) begin : TREES
            decision_tree #(.DW(DW), .N_COEF(N_COEF), .TREE_ID(t)) u_tree (
                .feat_flat(feat_flat),
                .vote(votes[t])
            );
        end
    endgenerate

    function automatic integer popcount(input [NUM_TREES-1:0] v);
        integer j;
        begin
            popcount = 0;
            for (j = 0; j < NUM_TREES; j = j + 1) popcount = popcount + v[j];
        end
    endfunction

    assign chf_detected = (popcount(votes) * 2) > NUM_TREES;

endmodule