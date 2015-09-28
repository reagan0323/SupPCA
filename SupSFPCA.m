function [B,V,U,se2,Sf]=SupSFPCA(Y,X,r,paramstruct)

% Created: 2014.2.12
% By: Gen Li

% tunings selection is embedded 
% (tuning range is automatically determined)
% modify the sequential opt problem a bit to be more rigorous (change Xs)
% normalize v in EACH iteration
% Sf is diagonal in each iteration
% add sparsity to each layer of B
% use Matlab LASSO solver
% customize tuning, eg. struct('lambda',0) to eliminate v sparsity 
% center X, Y before using SupSFPC
% default Omega is only valid for even spaced obs!!!

% allow customized Omega (e.g., in yield case)

ind_lam=1; % default: v sparse
ind_alp=1; % default: v smooth
ind_gam=1; % default: b sparse
ind_Omg=1; % default: Omega even spaced
start=0; % default: SVD initial est
max_niter=1e3; % max num of iter for EM
convg_thres=1E-6;  % EM convergence rule
vmax_niter=1e2; % max num of iter for each v_k
vconvg_thres=1E-4; % v_k convergence rule

if nargin > 3 ;   %  then paramstruct is an argument
  if isfield(paramstruct,'lambda') ;    %  then change to input value
    ind_lam = getfield(paramstruct,'lambda') ; 
  end ;
  if isfield(paramstruct,'alpha') ;    %  then change to input value
    ind_alp = getfield(paramstruct,'alpha') ; 
  end ;
  if isfield(paramstruct,'gamma') ;    %  then change to input value
    ind_gam = getfield(paramstruct,'gamma') ; 
  end ;
  if isfield(paramstruct,'start') ;    %  then change to input value
    start = getfield(paramstruct,'start') ; 
  end ; 
  if isfield(paramstruct,'Omega') ;    %  then change to input value
    Omega = getfield(paramstruct,'Omega') ; 
    ind_Omg=0;
  end ;
  if isfield(paramstruct,'convg_thres') ;    %  then change to input value
    convg_thres = getfield(paramstruct,'convg_thres') ; 
  end ; 
  if isfield(paramstruct,'vconvg_thres') ;    %  then change to input value
    vconvg_thres = getfield(paramstruct,'vconvg_thres') ; 
  end ; 

end;


[n,p]=size(X);
[n1,q]=size(Y);
if(rank(Y)<q)
    error('Do not run this code! Change initial and BIC df....');
    error('gamma cannot be set to zero!');
end;

% Pre-Check
if (n~=n1)
    error('X does not match Y! exit...');
elseif (rank(Y)~=q)
    error('Columns of Y are linearly dependent! exit...');
elseif (r>n || r>p)
    error('Too greedy on ranks! exit...');
end;

% set Omega
if(ind_Omg==1)
    Q=eye(p)*(-2);
    Q=spdiags(ones(p,1),1,Q);
    Q=spdiags(ones(p,1),-1,Q);
    Q=Q(:,2:(end-1));
    R=eye(p-2)*(2/3);
    R=spdiags(ones(p-2,1)*(1/6),1,R);
    R=spdiags(ones(p-2,1)*(1/6),-1,R);
    Omega=Q*inv(R)*Q'; % p*p
end;
oeig=eigs(Omega,1); % largest eig value of Omega


% initial est
if(start==0)
    [U,D,V]=svds(X,r); % initial V
    U=U*D; % initial U
    E=X-U*V';
    se2=var(E(:)); % initial se2
    B=inv(Y'*Y)*Y'*U; % initial B
    Sf=diag(diag((1/n)*(U-Y*B)'*(U-Y*B))); % initial Sf
    clear E D;
elseif(start==1)
    [B,V,U,se2,Sf]=EMS6(Y,X,r); % try to use SupSVD as initial estimation
end;



diff=1; % EM criterion
niter=0; % number of EM iter
while (niter<=max_niter && diff>convg_thres )
    % record last iter
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
    % estimate B and Sf
    if(ind_gam~=0)
        for k=1:r
            % Attention: lasso fcn always center X, Y, and return a
            % separate column of intercept; by default, it standardize each
            % column of X
            % Therefore, when using lasso, we always center the columns of
            % X and Y to avoid the trouble of intercept
            [SpParam,FitInfo]=lasso(Y,cond_Mean(:,k),'LambdaRatio',0,'Standardize',false); 
            BIC_score=n*log(FitInfo.MSE)+log(n)*FitInfo.DF;
            [~,ind]=min(BIC_score);
%             figure(1);clf;plot(FitInfo.Lambda,BIC_score);
%             figure(2);clf;plot(FitInfo.Lambda,SpParam);
            B(:,k)=SpParam(:,ind);
        end;
    else % if gamma=0, no B-sparsity
        B=inv(Y'*Y)*Y'*cond_Mean;
    end;
    %
    % estimate Sf
    Sf=diag(diag( (cond_quad + (Y*B)'*(Y*B)- (Y*B)'*cond_Mean- cond_Mean'*(Y*B) )/n )); % r*r
    %
    % estimate V
    for k=1:r % kth layer
        % some critical values
        theta=X'*cond_Mean(:,k)-(V_old*cond_quad(:,k)-V_old(:,k)*cond_quad(k,k)); % p*1
        c=cond_quad(k,k); % E(u_k'u_k|X), 1*1
        
        % select smooth tuning (LOOCV w/o refitting)
        if(ind_alp~=0)
            alphavec=0:0.1:10; % smooth tuning range
            cv_score=zeros(size(alphavec));
            for ialp=1:length(alphavec)
                alpha=alphavec(ialp);
                hat=inv(eye(p)+alpha*Omega);
                vest=hat*theta/c;
                cv_score(ialp)=(1/p)*sum(((theta/c-vest)./(1-diag(hat))).^2);
            end;
            [~,I]=min(cv_score);
            optalp=alphavec(I); % optimal alpha for this iteration
%           figure(k);clf;plot(alphavec,cv_score,'.-');title('Smoothness Tuning');
        else % no V-smoothness
            optalp=0;
        end;
 
        % specify sparsity tuning (for gaussian error)
        if(ind_lam~=0)
            optlam=sqrt(2*log(p)*se2_old/c);               
        else % no V sparsity
            optlam=0;
        end;
        
        
        L=1+optalp*oeig; 
        vk_old=V_old(:,k); % initial value for v_k is from last EM iteration
        vdiff=1;
        vniter=0;
        while(vniter<=vmax_niter && vdiff>vconvg_thres ) % iteration for estimating v_k
            df= -theta/c + (eye(p)+optalp*Omega)*vk_old;
            vk=soft_thr(vk_old-(1/L)*df, optlam/L);
            % set norm=1
            if norm(vk)==0
                warning('zero vector v!');
            else
                vk=vk/norm(vk);
            end;
            vdiff=norm(vk-vk_old)^2;
            vk_old=vk;
            vniter=vniter+1;
        end;
        V(:,k)=vk;
%         figure(k);clf;plot(vk);
    end;
    %         
    % Estimate se2
    se2=(trace(X*(X'-2*V*cond_Mean')) + n*trace(V'*V*cond_Var) + trace((cond_Mean'*cond_Mean)*(V'*V)))/(n*p);

        
    % stopping rule
    diff=norm(V-V_old,'fro')^2;
    niter=niter+1;

end;


% Print convergence information
if niter<max_niter
    disp(['SupFPCA converges at precision ',num2str(convg_thres),' after ',num2str(niter),' iterations.']);
else
    disp(['SupFPCA NOT converge at precision ',num2str(convg_thres),' after ',num2str(max_niter),' iterations!!!']);
end;



% reorder V and others
[~,I]=sort(diag(V'*X'*X*V),'descend');
V=V(:,I);
B=B(:,I);
Sf=Sf(I,I);


% output U
Sfinv=inv(Sf);
weight=inv(eye(r)+se2*Sfinv); % r*r
U=(se2*Y*B*Sfinv + X*V)*weight;

end

function out = soft_thr(in,lambda)
    out = sign(in).*max(abs(in) - lambda,0);
end