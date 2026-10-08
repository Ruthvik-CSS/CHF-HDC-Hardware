// tb_chf.v
// PURE RTL VERIFICATION - Works with Icarus Verilog (no break, no empty labels)

`include "hdc_params.vh"

module tb_chf;

    // ========================================================================
    // Clock and Reset
    // ========================================================================
    reg clk;
    reg rst_n;

    // ========================================================================
    // DUT Interfaces
    // ========================================================================
    reg                           sample_valid;
    reg  signed [15:0]            sample_in;
    wire                          features_valid;
    wire [14*16-1:0]              coeff_out_flat;

    reg                           start_classification;
    wire                          result_valid;
    wire                          result_chf;

    // ========================================================================
    // DUT Instantiations
    // ========================================================================
    wavelet_ca4 u_fe (
        .clk(clk),
        .rst_n(rst_n),
        .sample_valid(sample_valid),
        .sample_in(sample_in),
        .features_valid(features_valid),
        .coeff_out_flat(coeff_out_flat)
    );

    wire signed [15:0] feat_in [0:13];
    genvar gf;
    generate
        for (gf = 0; gf < 14; gf = gf + 1) begin : FEAT_EXTRACT
            assign feat_in[gf] = coeff_out_flat[(gf+1)*16-1 -: 16];
        end
    endgenerate

    ensemble_top u_classifier (
        .clk(clk),
        .rst_n(rst_n),
        .start(start_classification),
        .feat_in(feat_in),
        .result_valid(result_valid),
        .result_chf(result_chf)
    );

    // ========================================================================
    // Clock Generation
    // ========================================================================
    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    // ========================================================================
    // Testbench Control
    // ========================================================================
    integer ecg_file;
    integer labels_file;
    integer log_file;
    integer bytes_read;
    integer scan_ret;
    integer window_count;
    integer total_windows;
    integer correct_count;
    integer mismatch_count;
    integer i;
    integer max_windows;

    // Confusion-matrix counters (golden_label==1 / result_chf==1 treated as
    // the positive/CHF class):
    //   TP: golden=1, RTL=1   FN: golden=1, RTL=0
    //   TN: golden=0, RTL=0   FP: golden=0, RTL=1
    integer tp_count;
    integer tn_count;
    integer fp_count;
    integer fn_count;
    real    sensitivity;   // TP / (TP + FN)  -- true positive rate, recall
    real    specificity;   // TN / (TN + FP)  -- true negative rate

    localparam PAD_CYCLES = 6; // DRIVE_CYCLES(256) - WINDOW_CYCLES(250) = 6

    reg [7:0] buffer_ptr;

    reg signed [15:0] ecg_buffer [0:249];

    reg golden_label;
    reg rtl_prediction;
    reg classification_started;

    // ========================================================================
    // Main Test Process
    // ========================================================================
    initial begin
        // Initialize
        rst_n = 0;
        sample_valid = 0;
        sample_in = 0;
        start_classification = 0;
        classification_started = 0;
        window_count = 0;
        total_windows = 0;
        correct_count = 0;
        mismatch_count = 0;
        tp_count = 0;
        tn_count = 0;
        fp_count = 0;
        fn_count = 0;

        // Runtime-configurable window cap: run e.g.
        //   vvp sim.vvp +MAX_WINDOWS=2000
        // to test on a subset without recompiling or touching
        // ecg_samples.bin/labels.txt. Defaults to "all of them" if not given.
        if (!$value$plusargs("MAX_WINDOWS=%d", max_windows)) begin
            max_windows = 10000;
        end
        $display("Running up to %0d windows (pass +MAX_WINDOWS=N to vvp to change)", max_windows);

        // ====================================================================
        // Open Files
        // ====================================================================
        ecg_file = $fopen("ecg_samples.bin", "rb");
        if (ecg_file == 0) begin
            $display("ERROR: Could not open ecg_samples.bin");
            $finish;
        end

        labels_file = $fopen("labels.txt", "r");
        if (labels_file == 0) begin
            $display("ERROR: Could not open labels.txt");
            $finish;
        end

        log_file = $fopen("results/simulation_results.txt", "w");
        if (log_file == 0) begin
            $display("WARNING: Could not open log file (did you 'mkdir results'?)");
        end

        // ====================================================================
        // Start Simulation
        // ====================================================================
        $display("\n");
        $display("========================================================");
        $display("           FULL RTL PIPELINE VERIFICATION");
        $display("      ECG -> DWT -> Quantizer -> HDC Classifier");
        $display("========================================================");
        $display("\n");

        // Write header to log file
        $fdisplay(log_file, "RTL Verification Results");
        $fdisplay(log_file, "=========================");
        $fdisplay(log_file, "Window | RTL Output | Ground Truth | Status");
        $fdisplay(log_file, "-------|-------------|--------------|--------");

        // Apply reset (negedge-aligned so it's clean before our negedge-driven stimulus starts)
        repeat (20) @(negedge clk);
        rst_n = 1;
        $display("System reset complete, starting verification...\n");

        // ====================================================================
        // Main Processing Loop - Using repeat with max windows
        // ====================================================================
        begin : MAIN_LOOP
            for (window_count = 0; window_count < max_windows; window_count = window_count + 1) begin

                // ----------------------------------------------------------------
                // Step 1: Read one window (250 samples)
                // ----------------------------------------------------------------
                buffer_ptr = 0;
                while (buffer_ptr < 250) begin
                    bytes_read = $fread(sample_in, ecg_file);
                    if (bytes_read == 2) begin
                        ecg_buffer[buffer_ptr] = sample_in;
                        buffer_ptr = buffer_ptr + 1;
                    end else begin
                        disable MAIN_LOOP;
                    end
                end

                // ----------------------------------------------------------------
                // Step 2: Read ground truth label
                // ----------------------------------------------------------------
                scan_ret = $fscanf(labels_file, "%d\n", golden_label);
                if (scan_ret != 1) begin
                    disable MAIN_LOOP; 
                end

                // ----------------------------------------------------------------
                // Step 3: Feed ECG samples to FE block (negedge-driven: no race
                // with wavelet_ca4's own posedge-triggered logic)
                // ----------------------------------------------------------------
                for (i = 0; i < 250; i = i + 1) begin
                    @(negedge clk);
                    sample_valid = 1;
                    sample_in = ecg_buffer[i];
                end

                // Edge-pad with the last real sample for PAD_CYCLES more cycles
                // so the FE pipeline can fully drain
                for (i = 0; i < PAD_CYCLES; i = i + 1) begin
                    @(negedge clk);
                    sample_valid = 1;
                    sample_in = ecg_buffer[249];
                end

                @(negedge clk);
                sample_valid = 0;

                // Wait for the completed feature set
                while (!features_valid) begin
                    @(posedge clk);
                end
                @(posedge clk);

                // ----------------------------------------------------------------
                // Step 4: Start classification (negedge-driven, same reasoning)
                // ----------------------------------------------------------------
                @(negedge clk);
                start_classification = 1;
                classification_started = 1;
                @(negedge clk);
                start_classification = 0;

                // ----------------------------------------------------------------
                // Step 5: Wait for result
                // ----------------------------------------------------------------
                while (!result_valid) begin
                    @(posedge clk);
                end

                rtl_prediction = result_chf;
                classification_started = 0;

                // ----------------------------------------------------------------
                // Step 6: Compare with ground truth
                // ----------------------------------------------------------------
                total_windows = total_windows + 1;

                if (rtl_prediction == golden_label) begin
                    correct_count = correct_count + 1;
                    $display("Window %4d: CORRECT   RTL=%0d  Truth=%0d",
                             window_count, rtl_prediction, golden_label);
                    $fdisplay(log_file, "%6d | %11d | %12d |   PASS",
                             window_count, rtl_prediction, golden_label);
                end else begin
                    mismatch_count = mismatch_count + 1;
                    $display("Window %4d: MISMATCH  RTL=%0d  Truth=%0d",
                             window_count, rtl_prediction, golden_label);
                    $fdisplay(log_file, "%6d | %11d | %12d |   FAIL",
                             window_count, rtl_prediction, golden_label);
                end

                // Confusion-matrix tally
                if (golden_label == 1'b1) begin
                    if (rtl_prediction == 1'b1)
                        tp_count = tp_count + 1;
                    else
                        fn_count = fn_count + 1;
                end else begin
                    if (rtl_prediction == 1'b1)
                        fp_count = fp_count + 1;
                    else
                        tn_count = tn_count + 1;
                end

                if ((window_count % 100) == 0 && window_count > 0) begin
                    $display("\n  Progress: %0d windows processed", window_count + 1);
                    $display("  Accuracy so far: %0.2f%% (%0d/%0d)\n",
                             (correct_count * 100.0) / total_windows,
                             correct_count, total_windows);
                end

                repeat (5) @(negedge clk);
            end
        end

        // ====================================================================
        // Simulation Complete - Generate Report
        // ====================================================================
        $fclose(ecg_file);
        $fclose(labels_file);

        $display("\n");
        $display("========================================================");
        $display("                 VERIFICATION COMPLETE");
        $display("========================================================");
        $display("");
        $display("  Total Windows:  %8d", total_windows);
        $display("  Correct:        %8d", correct_count);
        $display("  Mismatches:     %8d", mismatch_count);
        if (total_windows > 0) begin
            $display("  RTL Accuracy:   %8.2f%%",
                     (correct_count * 100.0) / total_windows);
        end
        $display("");
        $display("  Confusion Matrix (positive class = CHF, label/result == 1)");
        $display("    TP: %6d   FN: %6d", tp_count, fn_count);
        $display("    FP: %6d   TN: %6d", fp_count, tn_count);

        if ((tp_count + fn_count) > 0) begin
            sensitivity = (tp_count * 100.0) / (tp_count + fn_count);
            $display("  Sensitivity (TPR, recall): %6.2f%%  (%0d/%0d)",
                     sensitivity, tp_count, tp_count + fn_count);
        end else begin
            sensitivity = 0.0;
            $display("  Sensitivity (TPR, recall): N/A (no positive-class windows)");
        end

        if ((tn_count + fp_count) > 0) begin
            specificity = (tn_count * 100.0) / (tn_count + fp_count);
            $display("  Specificity (TNR):         %6.2f%%  (%0d/%0d)",
                     specificity, tn_count, tn_count + fp_count);
        end else begin
            specificity = 0.0;
            $display("  Specificity (TNR):         N/A (no negative-class windows)");
        end
        $display("========================================================");

        $fdisplay(log_file, "");
        $fdisplay(log_file, "Confusion Matrix (positive class = CHF, label/result == 1)");
        $fdisplay(log_file, "  TP: %0d   FN: %0d   FP: %0d   TN: %0d",
                  tp_count, fn_count, fp_count, tn_count);
        if ((tp_count + fn_count) > 0)
            $fdisplay(log_file, "  Sensitivity (TPR): %0.2f%%", sensitivity);
        else
            $fdisplay(log_file, "  Sensitivity (TPR): N/A");
        if ((tn_count + fp_count) > 0)
            $fdisplay(log_file, "  Specificity (TNR): %0.2f%%", specificity);
        else
            $fdisplay(log_file, "  Specificity (TNR): N/A");

        if (total_windows > 0) begin
            if (mismatch_count == 0) begin
                $display("\n  ALL TESTS PASSED - READY FOR SYNTHESIS!");
            end else begin
                $display("\n  VERIFICATION COMPLETE - REVIEW MISMATCHES");
            end
        end else begin
            $display("\n  WARNING: 0 windows processed -- check ecg_samples.bin/labels.txt");
        end

        $fclose(log_file);

        $display("\nDetailed results saved to: results/simulation_results.txt");
        $finish;
    end

    // ========================================================================
    // Safety Timeout
    // ========================================================================
    reg [31:0] timeout_counter;
    always @(posedge clk) begin
        if (rst_n && classification_started && !result_valid) begin
            timeout_counter <= timeout_counter + 1;
            if (timeout_counter > 1000000) begin 
                $display("ERROR: Classification timeout at window %0d", window_count);
                $finish;
            end
        end else begin
            timeout_counter <= 0;
        end
    end

endmodule