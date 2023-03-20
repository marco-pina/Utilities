 function table(title,headers,labels,values,label_width,val_width, ... 
                val_precis) 
% 3  
% 4 % Copyright (C) 2002 Dynare Team 
% 5 % 
% 6 % This file is part of Dynare. 
% 7 % 
% 8 % Dynare is free software: you can redistribute it and/or modify 
% 9 % it under the terms of the GNU General Public License as published by 
% 10 % the Free Software Foundation, either version 3 of the License, or 
% 11 % (at your option) any later version. 
% 12 % 
% 13 % Dynare is distributed in the hope that it will be useful, 
% 14 % but WITHOUT ANY WARRANTY; without even the implied warranty of 
% 15 % MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the 
% 16 % GNU General Public License for more details. 
% 17 % 
% 18 % You should have received a copy of the GNU General Public License 
% 19 % along with Dynare.  If not, see <http://www.gnu.org/licenses/>. 
 
   label_width = max(size(deblank(strvcat(headers(1,:),labels)),2)+2, ... 
                    label_width); 
   val_width = max(size(deblank(headers(2:end,:)),2)+2,val_width); 
  label_fmt = sprintf('%%-%ds',label_width); 
   header_fmt = sprintf('%%-%ds',val_width); 
   val_fmt = sprintf('%%%d.%df',val_width,val_precis); 
   if length(title) > 0 
     disp(sprintf('\n\n%s\n',title)); 
   end 
  if length(headers) > 0 
     hh = sprintf(label_fmt,headers(1,:)); 
    hh = [hh char(32*ones(1,floor(val_width/4)))]; 
     for i=2:size(headers,1) 
       hh = [hh sprintf(header_fmt,headers(i,:))]; 
     end 
     disp(hh); 
   end 
   for i=1:size(values,1) 
     disp([sprintf(label_fmt,labels(i,:)) sprintf(val_fmt,values(i,:))]); 
   end 
