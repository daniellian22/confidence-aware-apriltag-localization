function run_xy_motion_only_apriltag(foldername)
    % Motion-only scan for AprilTag video capture (no RSTD/mmWave commands).
    % Logs commanded positions + timestamps.
    
        % ---------------- User parameters ----------------
        % Windows
        arduino_port = 'COM13';
        % Mac example:
        % arduino_port = '/dev/cu.usbserial-110';
    
        x_steps   = 3;
        y_steps   = 20;
        spacing_x = 7.6;   % mm
        spacing_y = 1.0;   % mm
    
        home_x = 1.0;      % mm
        home_y = 1.0;      % mm
    
        serpentine = true; % set false to always scan y=1->y_steps
    
        % For video: small dwell helps tag detection at each stop
        dwell_s  = 0.20;   % time to remain at each point (no sensor recording)
        settle_s = 0.00;   % extra settling after moves, if needed
    
        % Output
        out_base = fullfile('C:\ti\data\', foldername);
        if ~exist(out_base, 'dir'); mkdir(out_base); end
        log_path = fullfile(out_base, 'cmd_log.mat');
    
        % ---------------- Serial init ----------------
        arduinoObj = serial(arduino_port, 'BaudRate', 115200);
        arduinoObj.Timeout = 30;
        fopen(arduinoObj);
        c = onCleanup(@() safe_close(arduinoObj));
    
        pause(2.0); % allow Arduino serial to settle
    
        % ---------------- Go home ----------------
        sent_success = tcp_handshake(arduinoObj, home_x);
        while ~sent_success
            pause(0.5);
            sent_success = tcp_handshake(arduinoObj, home_x);
        end
    
        sent_success = tcp_handshake(arduinoObj, 1000 + home_y);
        while ~sent_success
            pause(0.5);
            sent_success = tcp_handshake(arduinoObj, 1000 + home_y);
        end
    
        pause(1.0);
    
        % ---------------- Scan ----------------
        num_pts = x_steps * y_steps;
        % Log: [pt_idx, x_idx, y_idx, x_mm, y_mm, unix_time]
        CMD = zeros(num_pts, 6);
    
        pt = 0;
        curr_x = home_x;
        curr_y = home_y;
    
        for x_idx = 1:x_steps
            x_pos = home_x + (x_idx - 1) * spacing_x;
    
            % Move X once per column
            if abs(x_pos - curr_x) > 1e-6
                sent_success = tcp_handshake(arduinoObj, x_pos);
                while ~sent_success
                    pause(0.2);
                    sent_success = tcp_handshake(arduinoObj, x_pos);
                end
                curr_x = x_pos;
                pause(settle_s);
            end
    
            % Choose Y order
            if serpentine && mod(x_idx,2)==0
                y_range = y_steps:-1:1;
            else
                y_range = 1:y_steps;
            end
    
            for y_idx = y_range
                y_pos = home_y + (y_idx - 1) * spacing_y;
    
                if abs(y_pos - curr_y) > 1e-6
                    sent_success = tcp_handshake(arduinoObj, 1000 + y_pos);
                    while ~sent_success
                        pause(0.2);
                        sent_success = tcp_handshake(arduinoObj, 1000 + y_pos);
                    end
                    curr_y = y_pos;
                    pause(settle_s);
                end
    
                pt = pt + 1;
                CMD(pt,:) = [pt, x_idx, y_idx, x_pos, y_pos, posixtime(datetime('now'))];
    
                pause(dwell_s);
            end
        end
    
        % ---------------- Return home ----------------
        if abs(curr_y - home_y) > 1e-6
            tcp_handshake(arduinoObj, 1000 + home_y);
        end
        if abs(curr_x - home_x) > 1e-6
            tcp_handshake(arduinoObj, home_x);
        end
    
        save(log_path, 'CMD');
        disp(['Saved command log: ' log_path]);
    end
    
    function safe_close(arduinoObj)
        try
            if strcmp(arduinoObj.Status,'open'); fclose(arduinoObj); end
        catch
        end
        try
            delete(arduinoObj);
        catch
        end
    end
    
    function sent_success = tcp_handshake(arduinoObj, tx_msg)
        % Protocol: send tx_msg; Arduino replies tx_msg+123; send back tx_msg+123.
        flushinput(arduinoObj);
        fprintf(arduinoObj, '%g\n', tx_msg);
    
        rx_msg = fscanf(arduinoObj, '%f');
    
        if abs(rx_msg - (tx_msg + 123)) < 1e-2
            fprintf(arduinoObj, '%g\n', tx_msg + 123);
            sent_success = true;
        else
            sent_success = false;
        end
    end
    