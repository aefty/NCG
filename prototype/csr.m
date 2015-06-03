%% CSR Prototype -  Condensed Sparse Matrix code template
%% 2015 - LMST Project
%% Aryan Eftekhari

classdef csr < handle

    properties (Access = 'public')
        rows;           % C++ -> int rows
        cols;           % C++ -> int cols
        val;            % C++ -> std::vector<double const> val
        col_ind;        % C++ -> std::vector<int const> col_ind
        std_format;     % For validation only
    end

    methods (Access = 'public')
        %Constructor
        function this = csr(flat,row_num,col_num)
            this.rows=row_num;
            this.cols=col_num;

            temp_val=0.0;
            index = 0;

            for row=0:this.rows-1                       % Zero Base
                for col=0:this.cols-1                   % Zero Base
                    index = row*col_num+col;
                    temp_val =flat(index+1);            % Matlab not 0 based

                    if(temp_val~=0.0 || col == 0 )
                        this.val(end+1)= temp_val;      % C++ -> val.push_back(temp)
                        this.col_ind(end+1)= col;       % C++ -> col_ind.push_back(col)
                    end
                end

            end
        end


        %For Validation only
        function this = inv(this)
            this.std_format = zeros(this.rows,this.cols);
            s = max(size(this.val));
            row=0;
            last_col=0;

            for i=1:s

                val = this.val(i);
                col = this.col_ind(i)+1;
                next_row = col ==1;

                if(next_row)
                    row=row+1;
                end

                this.std_format(row,col) = val;
            end
        end
    end
end