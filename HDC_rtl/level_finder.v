// level_finder.v
// Binary-search/sequential threshold comparator for quantization.

`include "hdc_params.vh"

module level_finder (
    input  wire                         clk,
    input  wire                         rst_n,
    input  wire                         start,
    input  wire signed [`THRESH_W-1:0]  feat_value,
    input  wire signed [`THRESH_W-1:0]  thresh_in,
    output reg  [4:0]                   thresh_idx,
    output reg                          done,
    output reg  [4:0]                   level_idx,

    output wire                         thresh_rd_en
);

    // State machine: sequential threshold comparison
    // Since N_LEVELS=20, compare against 19 thresholds sequentially
    localparam IDLE       = 3'd0,
               WAIT_ROM   = 3'd1,   // Wait for thresh_in to be valid
               COMPARE    = 3'd2,
               DONE_ST = 3'd3;
    reg [2:0] state;
    reg [4:0] count;      // 0..N_LEVELS-2 (19 thresholds)
    reg [4:0] idx;           // 0..N_LEVELS-1 (20 levels)

    assign thresh_rd_en = (state == WAIT_ROM);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            done <= 1'b0;
            thresh_idx <= 5'd0;
            level_idx <= 5'd0;
            idx <= 5'd0;
            count <= 5'd0;
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        thresh_idx <= 5'd0;
                        idx        <= 5'd0;
                        count      <= 5'd0;
                        state      <= WAIT_ROM;
                    end
                end

                WAIT_ROM: begin
                    // Wait one cycle for thresh_in to be valid from ROM
                    state <= COMPARE;
                end
                
                COMPARE: begin
                    // thresh_in now correctly corresponds to thresh_idx==idx
                    if (feat_value > thresh_in)
                        count <= count + 1'b1;

                    if (idx == `N_LEVELS - 2) begin
                        level_idx <= (feat_value > thresh_in) ? (count + 1'b1) : count;
                        state     <= DONE_ST;
                    end else begin
                        idx        <= idx + 1'b1;
                        thresh_idx <= idx + 1'b1;  // drive next address
                        state      <= WAIT_ROM;
                    end
                end

                DONE_ST: begin
                    done  <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule