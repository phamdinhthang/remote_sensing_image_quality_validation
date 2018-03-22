%low pass filtering with input sigma
function lowFreq_part = lowPassFilter(originalPart,sigma)
%gaussian filter for low pass filtering of the PRNU
Filter=GaussianFilter(sigma);
%process the PRNU border
L=floor(length(Filter)/2);
originalPart_extended = [ones(1,L) originalPart ones(1,L)];
lowFreq_part = conv(originalPart_extended,Filter);
lowFreq_part = lowFreq_part(2*L+1:end-2*L);