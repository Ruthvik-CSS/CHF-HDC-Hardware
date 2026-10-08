// hdc_core.v

`include "hdc_params.vh"

module hdc_core (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        start,
    input  wire [1:0]  member_id,
    input  wire [4:0]  level_idx [0:`N_FEAT-1],

    // ROM interface
    output reg  [1:0]  rom_pos_member,
    output reg  [3:0]  rom_pos_feat,
    output reg  [9:0]  rom_pos_bit_idx,
    input  wire        rom_pos_bit_out,

    output reg  [1:0]  rom_lvl_member,
    output reg  [4:0]  rom_lvl_level,
    output reg  [9:0]  rom_lvl_bit_idx,
    input  wire        rom_lvl_bit_out,

    output reg  [1:0]  rom_cls_member,
    output reg         rom_cls_class,
    output reg  [9:0]  rom_cls_bit_idx,
    input  wire [7:0]  rom_cls_value_out,

   
    output wire         rom_pos_rd_en,
    output wire         rom_lvl_rd_en,
    output wire         rom_cls_rd_en,

    output reg         done,
    output reg         member_decision
);

    // State machine
    localparam IDLE        = 4'd0,
               FEAT_SET     = 4'd1,
               FEAT_WAIT    = 4'd2,
               FEAT_READ    = 4'd3,   
               FEAT_ACC     = 4'd4,
               DIM_ENCODE   = 4'd5,
               CLASS_SET0   = 4'd6,
               CLASS_WAIT0  = 4'd7,
               CLASS_READ0  = 4'd8,
               CLASS_WAIT1  = 4'd9,
               CLASS_READ1  = 4'd10,
               SIM_ACCUM    = 4'd11,
               FINAL        = 4'd12;

    reg [3:0] state;
    reg [9:0] dim_i;          
    reg [3:0] feat_i;         
    reg signed [4:0] feat_sum;
    reg signed [`SIM_W-1:0] sim0, sim1;
    reg encoded_bit;

    
    reg pos_bit_r, lvl_bit_r;
    reg signed [7:0] class0_val, class1_val;

    
    assign rom_pos_rd_en = (state == FEAT_SET) || (state == FEAT_WAIT) ||
                           (state == FEAT_READ) || (state == FEAT_ACC);
    assign rom_lvl_rd_en = rom_pos_rd_en;

    assign rom_cls_rd_en = (state == DIM_ENCODE) || (state == CLASS_SET0) ||
                           (state == CLASS_WAIT0) || (state == CLASS_READ0) ||
                           (state == CLASS_WAIT1) || (state == CLASS_READ1);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE; done <= 1'b0; member_decision <= 1'b0;
            dim_i <= 10'd0; feat_i <= 4'd0; feat_sum <= 5'sd0;
            sim0 <= {`SIM_W{1'b0}}; sim1 <= {`SIM_W{1'b0}};
        end else begin
            case (state)
                IDLE: begin
                    done <= 1'b0;
                    if (start) begin
                        dim_i <= 10'd0;
                        sim0 <= {`SIM_W{1'b0}};
                        sim1 <= {`SIM_W{1'b0}};
                        feat_i <= 4'd0;
                        feat_sum <= 5'sd0;
                        state <= FEAT_SET;
                    end
                end

                FEAT_SET: begin
                    rom_pos_member  <= member_id;
                    rom_pos_feat    <= feat_i;
                    rom_pos_bit_idx <= dim_i;
                    rom_lvl_member  <= member_id;
                    rom_lvl_level   <= level_idx[feat_i];
                    rom_lvl_bit_idx <= dim_i;
                    state <= FEAT_WAIT;
                end

                FEAT_WAIT: begin
                    state <= FEAT_READ;
                end

                FEAT_READ: begin
                    pos_bit_r <= rom_pos_bit_out;
                    lvl_bit_r <= rom_lvl_bit_out;
                    state <= FEAT_ACC;
                end

                FEAT_ACC: begin
                    if (pos_bit_r == lvl_bit_r)
                        feat_sum <= feat_sum + 5'sd1;
                    else
                        feat_sum <= feat_sum - 5'sd1;

                    if (feat_i == `N_FEAT - 1) begin
                        state <= DIM_ENCODE;
                    end else begin
                        feat_i <= feat_i + 1'b1;
                        state <= FEAT_SET;
                    end
                end

                DIM_ENCODE: begin
                    encoded_bit <= (feat_sum >= 0);

                    rom_cls_member  <= member_id;
                    rom_cls_class   <= 1'b0;
                    rom_cls_bit_idx <= dim_i;
                    state <= CLASS_SET0;
                end

                CLASS_SET0: begin
                    state <= CLASS_WAIT0;
                end

                CLASS_WAIT0: begin
                    state <= CLASS_READ0;
                end

                CLASS_READ0: begin
                    class0_val <= rom_cls_value_out;
                    rom_cls_class <= 1'b1;
                    state <= CLASS_WAIT1;
                end

                CLASS_WAIT1: begin
                    state <= CLASS_READ1;
                end

                CLASS_READ1: begin
                    class1_val <= rom_cls_value_out;
                    state <= SIM_ACCUM;
                end

                SIM_ACCUM: begin
                    if (encoded_bit) begin
                        sim0 <= sim0 + class0_val;
                        sim1 <= sim1 + class1_val;
                    end else begin
                        sim0 <= sim0 - class0_val;
                        sim1 <= sim1 - class1_val;
                    end

                    if (dim_i == `DIM - 1) begin
                        state <= FINAL;
                    end else begin
                        dim_i <= dim_i + 1'b1;
                        feat_i <= 4'd0;
                        feat_sum <= 5'sd0;
                        state <= FEAT_SET;
                    end
                end

                FINAL: begin
                    member_decision <= (sim1 > sim0);
                    done <= 1'b1;
                    state <= IDLE;
                end
            endcase
        end
    end

endmodule