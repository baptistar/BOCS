% Author: Ricardo Baptista and Matthias Poloczek
% Date:   June 2018
%
% See LICENSE.md for copyright information
%

function pascal_row = pascal_row(row_idx)
% PASCAL_ROW: Function computes a row of the pascal triangle for sampling
% based on the l1-norm penalty

pascal_row = 1;

for i=0:row_idx-1
    new_item   = pascal_row(i+1)*(row_idx-i)/(i+1);
    pascal_row = [pascal_row, new_item]; 
end