function [B,V,U,se2,Sf]=SupPCA(Y,X,r)
% Updated version of EMS algorithm to fit the SupSVD model:
% X=UV' + E 
% U=YB + F 
% (or equiv X=YBV'+FV'+E) 
%
% Note: the only requirements are 
% 1) Y has full column rank
% 2) r<=min(n,p)
% 3) V(1) is positive
% 
% The identifiability condition is Sf is diagonal, V is orthonormal, B is
% unconstrained.
% The initial estimation is from SVD, and r can be larger than q
%
% Input
%       Y           n*q full column rank response matrix (necessarily n>=q)
%                   columns should be centered
%        
%       X           n*p (n may less than p) design matrix
%                   columns should be centered
%      
%       r           fixed rank of approxiamtion, r<=min(n,p)              
%
% Output
%       B           q*r coefficient matrix for U~Y, 
%                   
%       V           p*r coefficient matrix for X~U, with orthonormal columns
%                   for identifiability, V(1,:) always positive
%       se2         scalar, var(E)
%       Sf          r*r diagonal matrix, cov(F)
%
% Created: 2013.6.1
% By: Gen Li

[n,p]=size(X);
[n1,q]=size(Y);

% Pre-Check
if (n~=n1)
    error('X does not match Y! exit...');
elseif (r>min(n,p))
    error('Rank is too greedy! exit...');
elseif (rank(Y)~=q)
    error('Columns of Y are linearly dependent! exit...');
% elseif( abs(sum(mean(Y,1))+sum(mean(X,1)))>1E-2)
%     error('X/Y columns has not been centered! exit...');
end;


% % RRR naive start
% [H,~,~]=svds(X'*Y*(Y'*Y)^-.5,r);
% C=((Y'*Y)\Y')*X*H*H';
% [B_temp,D_temp,V]=svds(C,r);
% B=B_temp*D_temp;
% [temp1,temp2,temp3]=svds(X,r);
% temp4=X-temp1*temp2*temp3';
% se2=var(temp4(:));
% Sf=V'*cov(X-Y*C)*V-eye(r)*se2;
% [VV,DD]=eig(Sf);
% DD=diag(abs(diag(DD))); % make sure Sf is pd
% Sf=VV*DD*VV';
% [newV,newSf]=eig(V*Sf*V');
% Sf=newSf(1:r,1:r);
% B=B*V'*newV(:,1:r);
% V=newV(:,1:r);

% SVD start
[U,D,V]=svds(X,r);
U=U*D;
E=X-U*V';
se2=var(E(:));
B=inv(Y'*Y)*Y'*U;
Sf=diag(diag((1/n)*(U-Y*B)'*(U-Y*B)));

temp1=X*V-Y*B; % n*r
temp2=X-Y*B*V'; % n*p
logl=(-n/2)*(log(det(Sf+se2*eye(r)))+(p-r)*log(se2))-...
         (.5/se2)*trace(temp2*temp2')-...
         .5*trace((temp1'*temp1)/(Sf+se2*eye(r)))+...
         (.5/se2)*trace(temp1'*temp1);
rec=[logl];

max_niter=1E5;
convg_thres=1E-6;  % 1E-5 is stringent; 1E-2 is too loose
Ldiff=1;
Pdiff=1;
niter=0;

while (niter<=max_niter && (Pdiff>convg_thres))% || Pdiff>convg_thres) )
    % record last iter
    logl_old=logl;
    se2_old=se2;
    Sf_old=Sf;
    V_old=V;
    B_old=B;
    
    % E step
    % some critical values
    Sfinv=inv(Sf);
    weight=inv(eye(r)+se2*Sfinv); % r*r
    cond_Mean=(se2*Y*B*Sfinv + X*V)*weight; % E(U|X), n*r
    cond_Var=Sf*(eye(r)-weight); % cov(U(i)|X), r*r
    cond_quad=n*cond_Var + cond_Mean'*cond_Mean; % E(U'U|X), r*r
    
    % M step
    V=X'*cond_Mean/(cond_quad); % p*r
    se2=(trace(X*(X'-2*V*cond_Mean')) + n*trace(V'*V*cond_Var) + trace(cond_Mean*V'*V*cond_Mean'))/(n*p);
    B=(Y'*Y)\Y'*cond_Mean; % q*r
    Sf=(cond_quad + (Y*B)'*(Y*B)- (Y*B)'*cond_Mean- cond_Mean'*(Y*B) )/n; % r*r
    
    % S step (need to consider hat{Sf}=0 cases.....)
%     r_=rank(B); % r_<=min(r,q), the # of non-zero columns of B
%     V_=V;
%     [B_temp,D_temp,V]=svds(B*V_',r_);
%     B=[B_temp*D_temp,zeros(q,r-r_)]; % final B
%     if (r_<r)
%         disp('B has zero columns.');
%         V_comp=V_-V*(V'*V_); % project V_ to orth subsp of V, should be rank r-r_
%         [V_triangle,~,~]=svds(V_comp,r-r_);
%         [Q,~]=eig(V_triangle'*V_*Sf*V_'*V_triangle); 
%         V=[V,V_triangle*Q]; % final V
%     end;
%     Sf=V'*V_*Sf*V_'*V; % final Sf
    [newV,newSf,~]=svds(V*Sf*V',r);
    Sf=newSf(1:r,1:r);
    B=B*V'*newV(:,1:r);
    V=newV(:,1:r);
    
    % log likelihood
    temp1=X*V-Y*B; % n*r
    temp2=X-Y*B*V'; % n*p
    logl=(-n/2)*(log(det(Sf+se2*eye(r)))+(p-r)*log(se2))-...
         (.5/se2)*trace(temp2*temp2')-...
         .5*trace((temp1'*temp1)/(Sf+se2*eye(r)))+...
         (.5/se2)*trace(temp1'*temp1);
    rec=[rec,logl];
    
%     disp(['Iteration ',num2str(niter),', LogL=',num2str(logl),', Sf=',num2str(Sf),', norm(B)=',num2str(norm(B))]);
    
    % iteration termination
    Ldiff=logl-logl_old; % should be positive
%     Pdiff=abs(se2-se2_old)+sum(abs(Sf_old(:)-Sf(:)))+sum(abs(V_old(:)-V(:)))+sum(abs(B_old(:)-B(:)));
    Pdiff=norm(V-V_old,'fro')^2;
    niter=niter+1;

end;

if niter<max_niter
    disp(['EMS converges at precision ',num2str(convg_thres),' after ',num2str(niter),' iterations.']);
else
    disp(['EMS NOT converge at precision ',num2str(convg_thres),' after ',num2str(max_niter),' iterations!!!']);
end;

% plot(rec,'o-');
% ylabel('log likelihood');
% rec(11:15)/10000

% re-order V, and correspondingly B and Sf, U (one simple remedy for the improper order of V)
[~,I]=sort(std(X*V),'descend');
V=V(:,I);
B=B(:,I);
Sf=Sf(I,I);

% correct sign of V for identifiability
% also correct B and Sf
signchange=sign(V(1,:));
V=bsxfun(@times,V,signchange);
B=bsxfun(@times,B,signchange);
Sf=diag(signchange)*Sf*diag(signchange);

% output U
Sfinv=inv(Sf);
weight=inv(eye(r)+se2*Sfinv); % r*r
U=(se2*Y*B*Sfinv + X*V)*weight;


