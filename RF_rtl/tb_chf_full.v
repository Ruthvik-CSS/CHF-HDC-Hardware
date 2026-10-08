// =============================================================================
// tb_chf_top_accuracy.v
// =============================================================================
`timescale 1ns/1ps

module tb_chf_top_accuracy;
    localparam DW           = 16;
    localparam ACC_W        = 26;
    localparam N_COEF       = 14;
    localparam WIN_LEN      = 250;
    localparam DRIVE_CYCLES = 16*(N_COEF-1) + 48;

    parameter integer MAX_WINDOWS = 872000;
    localparam integer MAX_SAMPLES = MAX_WINDOWS * DRIVE_CYCLES;

    reg clk = 0;
    reg rst_n = 0;
    reg sample_valid = 0;
    reg signed [DW-1:0] sample_in = 0;

    wire classification_valid;
    wire chf_detected;
    wire [N_COEF*DW-1:0] feature_vector_flat;

    chf_top #(.DW(DW), .ACC_W(ACC_W), .N_COEF(N_COEF), .NUM_TREES(3)) dut (
        .clk(clk), .rst_n(rst_n),
        .sample_valid(sample_valid), .sample_in(sample_in),
        .classification_valid(classification_valid),
        .chf_detected(chf_detected),
        .feature_vector_flat(feature_vector_flat)
    );

    always #5 clk = ~clk;

    reg signed [DW-1:0] mem [0:MAX_SAMPLES-1];
    integer labels [0:MAX_WINDOWS-1];          

    integer total_windows;
    integer lfd, r, code, dummy;
    integer w, k;
    integer total, tp, tn, fp, fn;
    real acc, sens, spec;
    reg [8*256-1:0] hexfile, labelfile;

    initial begin
        if (!$value$plusargs("hexfile=%s", hexfile))     hexfile   = "ecg_samples.hex";
        if (!$value$plusargs("labelfile=%s", labelfile)) labelfile = "ecg_labels.txt";

        lfd = $fopen(labelfile, "r");
        if (lfd == 0) begin
            $display("ERROR: could not open label file '%0s'.", labelfile);
            $display("Run export_ecg_hex.py first, or pass +hexfile=... +labelfile=...");
            $finish;
        end

        r = $fgetc(lfd);
        if (r == "#") begin
            while (r != "\n" && r != -1) r = $fgetc(lfd);
        end else begin
            dummy = $ungetc(r, lfd);
        end

        total_windows = 0;
        while (!$feof(lfd)) begin
            code = $fscanf(lfd, "%d", labels[total_windows]);
            if (code == 1) begin
                total_windows = total_windows + 1;
                if (total_windows >= MAX_WINDOWS) begin
                    $display("ERROR: label file has more than MAX_WINDOWS=%0d windows.", MAX_WINDOWS);
                    $display("Recompile with -PMAX_WINDOWS=<larger number>.");
                    $finish;
                end
            end
        end
        $fclose(lfd);

        if (total_windows == 0) begin
            $display("ERROR: no labels read from '%0s'.", labelfile);
            $finish;
        end

        $display("Loading %0d windows (%0d samples) from %0s ...",
                  total_windows, total_windows*DRIVE_CYCLES, hexfile);
        $readmemh(hexfile, mem, 0, total_windows*DRIVE_CYCLES - 1);

        $display("Loaded %0d windows (%0d samples) from %0s / %0s",
                  total_windows, total_windows*DRIVE_CYCLES, hexfile, labelfile);

        rst_n = 0; sample_valid = 0; sample_in = 0;
        repeat (3) @(posedge clk);
        rst_n = 1;
        @(posedge clk);

        total = 0; tp = 0; tn = 0; fp = 0; fn = 0;

        for (w = 0; w < total_windows; w = w + 1) begin
            for (k = 0; k < DRIVE_CYCLES; k = k + 1) begin
                @(negedge clk);
                sample_valid = 1;
                sample_in    = mem[w*DRIVE_CYCLES + k];
            end
            @(negedge clk);
            sample_valid = 0;

            @(posedge clk);
            #1;

            if (!classification_valid) begin
                $display("WINDOW %0d: ERROR - classification_valid not asserted", w);
            end else begin
                total = total + 1;
                if (labels[w] == 1 && chf_detected == 1) tp = tp + 1;
                else if (labels[w] == 0 && chf_detected == 0) tn = tn + 1;
                else if (labels[w] == 0 && chf_detected == 1) fp = fp + 1;
                else if (labels[w] == 1 && chf_detected == 0) fn = fn + 1;
            end

            if ((w % 500) == 0 && w > 0)
                $display("  ... window %0d/%0d", w, total_windows);
        end

        if (total == 0) begin
            $display("ERROR: no windows were successfully classified.");
            $finish;
        end

        acc  = (tp + tn) * 100.0 / total;
        sens = (tp + fn) > 0 ? (tp * 100.0 / (tp + fn)) : 0.0;
        spec = (tn + fp) > 0 ? (tn * 100.0 / (tn + fp)) : 0.0;

        $display("\n=== Full-pipeline RTL accuracy (wavelet_ca4 + rf_classifier) on %0d windows ===", total);
        $display("Confusion matrix: TP=%0d FN=%0d TN=%0d FP=%0d", tp, fn, tn, fp);
        $display("Accuracy:    %0.2f%%", acc);
        $display("Sensitivity: %0.2f%%", sens);
        $display("Specificity: %0.2f%%", spec);
        $display("(paper reports 90.5%% / 93.01%% / 92.26%% on its own DS2 split)");
        $display("(compare against train_and_export.py's printed Python-side numbers on");
        $display(" the SAME records -- any large gap points at a mismatch between the");
        $display(" Python fixed-point feature extraction and wavelet_ca4.v's RTL.)");

        $finish;
    end
endmodule
