function SimulateImage = SimulateImage(InputImage,DS,PRNU,DNsigma)
%ham chen du lieu DS, PRNU vao Input Image
[M N] = size(InputImage);
SimulateImage= zeros(M,N,'uint8');

        if length(DS)==N && length(PRNU)==N
    
            DS = DS(:)'; PRNU=PRNU(:)';
            for i=1:M,
                SimulateImage(i,:)=uint8(PRNU.*double(InputImage(i,:)) + DS + DNsigma*randn(1,N));
            end

        else
            error('hang va cot phai bang DS va PRNU')
              
        end
            
        


        