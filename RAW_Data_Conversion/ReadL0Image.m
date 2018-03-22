function RawImage = ReadL0Image(bandName, imageDir, lmin, lmax)
% RawImage = ReadL0Image(bandName, imageDir, lmin, lmax)
% This function read a raw image from L0 directory
% The L0 filename is IMAGERY_[bandName].RAW
% lmin is the first line of the block to be read
% lmax is the last line of the block to be read
%
% Warning : use multibandread function

% Input parameters
if ~ismember(bandName,{'PAN','B1','B2','B3','B4'}), 
    error('The bandName is not valid');
end
imageFile=['IMAGERY_' bandName '.RAW'];
metadataFile='METADATA.L0';
if ~exist('imageDir','var') || isempty(imageDir), imageDir='.'; end
if ~exist('lmin','var') || isempty(lmin), lmin=1; end

% Read L0 properties
L0info=ReadL0Metadata(fullfile(imageDir,metadataFile));

if ~exist('lmax','var') || isempty(lmax), lmax = L0info.nRows; end

% Test lmax >=lmin
assert(lmax>=lmin,'lmax should be greater than lmin')

% Number of header pixels per line
NbPx_Header_PerLine=L0info.skipBytes_perLine/(L0info.nBits/8);

% Pixel type
if L0info.nBits==16 && strcmp(L0info.data_type,'UNSIGNED')
    pxType='uint16';
else
    error('NBits should be 16 et DataType Unsigned')
end

% Read the image block
RawImage = multibandread(fullfile(imageDir,imageFile),[L0info.nRows, L0info.nCols+NbPx_Header_PerLine, 1], [pxType '=>' pxType], 0, L0info.bandsLayout, L0info.byteOrder, ...
    {'Row', 'Range', [lmin,1,lmax]}, ...
    {'Col', 'Range', [NbPx_Header_PerLine+1,1,NbPx_Header_PerLine+ L0info.nCols]});

