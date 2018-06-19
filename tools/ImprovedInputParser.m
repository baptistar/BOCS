classdef ImprovedInputParser < inputParser
    %
    
    % This file is part of the Matlab Toolbox TENSALG for Tensor Algebra,
    % developed under the BSD Licence.
    % See the LICENSE file for conditions.
    
    properties
        unmatchedArgs
    end
    
    methods
        function p = ImprovedInputParser()
            p@inputParser();
            p.KeepUnmatched = true;
        end
        function parse(p,varargin)
            parse@inputParser(p,varargin{:});
            makeUnmatchedArgs(p);
        end
        
        function makeUnmatchedArgs(p)
            tmp = [fieldnames(p.Unmatched), ...
                struct2cell(p.Unmatched)];
            p.unmatchedArgs = reshape(tmp',[],1)';
        end
        
        function obj = passMatchedArgsToProperties(p,obj)
            fNames = fieldnames(p.Results);
            for k = 1:numel(fNames)
                obj.(fNames{k}) = p.Results.(fNames{k});
            end
        end
    end
end
