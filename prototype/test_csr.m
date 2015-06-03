%CSR Compression Test
clear; clc; close all;
N = 100;

for i =1:N
	% Create Sample - Matrix is a flattened  = horizontal ordered by [row1,row2,row3]
	A = rand(100);
	sparsity_index = i/N;
	A(A<sparsity_index)=0;

	sparse_sample=A';
	sparse_sample = sparse_sample(:)';

	s= size(A);

	A_csr = csr(sparse_sample,s(1),s(2));

	A_csr.inv;
	A_test = A_csr.std_format;

	if(sum(sum(A_test - A))==0)
		disp('Valid Compression');
	else
		warning(['INVALID CALCULATION']);
		A
		A_csr
		A_test = A_csr.std_format
		pause
	end

	sa = prod(size(A_csr.val));
	sb = prod(size(A_csr.col_ind));

	c = sum([sa,sb]) /prod(size(A))

	compression(i)=c;
end

figure;
plot(compression);
title('CSR Compression Test');
ylabel('Ratio  - Sparse Matrix Size / Full Matrix Size ');
xlabel('% Non-Zero Elements');
