function L0info = ReadL0Metadata(METADATA)
% L0info = ReadL0Metadata(METADATA)
%
% L0info.nRows	number of rows
% L0info.nCols	nomber of columns
% L0info.nBits	nombre of bits per pixel
% L0info.skipBytes_perLine	number of header bytes per line
% L0info.skipBytes  number of header bytes in the file
% L0info.byteOrder  byteOrder ('M' for big endian)
% L0info.data_type  'SIGNED' or 'UNSIGNED'

% Image parameters
if exist('METADATA','var') && ~isempty(METADATA)
    
    % Read the xml METADATA file
    tree = xml_read(METADATA);
    
    % Number of bands
    nBands=tree.Raster_Dimensions.NBANDS;
    % Number of columns
    nCols=tree.Raster_Dimensions.NCOLS;
    % Number of rows
    nRows=tree.Raster_Dimensions.NROWS;
    % Number of header bytes in the file
    skipBytes=tree.Data_Strip.BandParameters.BandParameter(1).SKIPBYTES;
    % Number of header bytes for each line
    skipBytes_perLine=tree.Data_Strip.BandParameters.BandParameter(1).SKIPBYTES_PER_LINE;
    
    % Number of bits per pixel
    nBits=tree.Raster_Encoding.NBITS;
    % ByteOrder
    if strcmp(tree.Raster_Encoding.BYTEORDER,'M')
        byteOrder='ieee-be';
    else if strcmp(tree.Raster_Encoding.BYTEORDER,'L')
            byteOrder='ieee-le';
        else
            disp('invalid byte order value !')
        end
    end
    
    % data type: unsigned, signed
    data_type=tree.Raster_Encoding.DATA_TYPE;
    % bands layout
    bandsLayout = tree.Raster_Encoding.BANDS_LAYOUT;
    % organisation of the bands
    data_file_organisation=tree.Data_Access.DATA_FILE_ORGANISATION;
    
    
else
    
    % default values
    skipBytes=0;
    skipBytes_perLine=872;
    nCols=7000;
    nRows=8238;
    nBands=1;
    nBits=16;
    byteOrder='ieee-le';
    data_type='UNSIGNED';
    bandsLayout = 'BSQ';
    data_file_organisation='BAND_SEPARATE';
    
end

if nBands > 1 && ~strcmp(data_file_organisation,'BAND_SEPARATE')
    error('Number of bands should be 1 and Bands should be separated'); 
else
    nBands=1;
end

L0info.nRows=nRows;
L0info.nCols=nCols;
L0info.nBits=nBits;
L0info.skipBytes_perLine=skipBytes_perLine;
L0info.skipBytes=skipBytes;
L0info.byteOrder=byteOrder;
L0info.data_type=data_type;
L0info.bandsLayout = bandsLayout;

