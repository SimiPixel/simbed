classdef controllerABC < handle 
    %   Interface for unrolling Matlab-controller in Python 
    %
    %   General logic is as follows: The `ctr` object in the following
    %   code is an instantiation of this class. In Python i will call this
    %   object's methods in the following manner.
    %   
    %   ctr.setSamplingRate(0.01)
    %   ctr.setTmax(5)
    %   ctr.setStateDimensions(4)
    %   ctr.preallocate()
    %
    %   rs = generate_reference()
    %   ctr.setReference(rs)   
    %
    %   for epiode in episodes:
    %       x,y = env.reset()
    %       
    %       ctr.setCurrentState(x)
    %       ctr.setCurrentObs(y)
    %       ctr.setStatus(0) <- This might be removed
    %
    %       ctr.before_episode()
    %       
    %       while not done:
    %           ctr.before_simulation_step()
    %
    %           u = ctr.getCurrentControl()
    %           x,y,rew,done = env.step(u)
    %           ctr.setCurrentState(x)
    %           ctr.setCurrentObs(y)
    %           ctr.setCurrentReward(rew)
    %           ctr.setStatus(1) <- This might be removed
    %   
    %           ctr.after_simulation_step()
    %   
    %       ctr.setStatus(2) <- This might be removed
    %
    %       ctr.after_episode()
    %       
    
    properties (SetAccess = private)
        x;
        xs;
        xss;

        y;
        ys;
        yss;
        
        r;
        rs;
        rss;
        
        u;
        us;
        uss;
        
        rew;
        rews;
        rewss;

        status; % 0|1|2 -> 0: Episode just started; 1: Episode is still going; 2: Episode just finished

        learning_rate;
        max_n_episodes;
        Ts;
        Tmax;
        state_dim;

        i_episode=1;
        i_time=1;
    end
    
    methods
        function obj = controllerABC(learning_rate, max_n_episodes)
            % Initialise controller
            obj.learning_rate = learning_rate;
            obj.max_n_episodes = max_n_episodes;
        end

        function before_episode(obj)
            % Reset time count
            obj.i_time=1;

            % Record initial state and initial observation
            obj.xs(obj.i_time,:) = obj.x;
            obj.ys(obj.i_time,:) = obj.y;

            % Set initial control 
            obj.u = 0.0;
        end

        function before_simulation_step(obj)
            % Update current reference
            obj.updateCurrentReference()

            % Update control u
            obj.u = obj.u - obj.learning_rate*(obj.r-obj.y);
        end

        function after_simulation_step(obj)
            % Record reward and control 
            obj.rews(obj.i_time,:) = obj.rew;
            obj.us(obj.i_time,:) = obj.u;

            % Increase time count
            obj.i_time = obj.i_time + 1;

            % Record state and observation
            obj.xs(obj.i_time,:) = obj.x;
            obj.ys(obj.i_time,:) = obj.y;
        end

        function after_episode(obj)
            % Record state over trials 
            obj.xss(obj.i_episode,:,:) = obj.xs;
            % Record observations
            obj.yss(obj.i_episode,:,:) = obj.ys;
            % Record control
            obj.uss(obj.i_episode,:,:) = obj.us;
            % Record rewards
            obj.rewss(obj.i_episode,:,:) = obj.rews;
            % Record references (if you want)
            obj.rss(obj.i_episode,:,:) = obj.rs;

             % Increase episode count
            obj.i_episode = obj.i_episode + 1;
        end

        function updateCurrentReference(obj)
            obj.r = obj.rs(obj.i_time);
        end

        %%%% Do *not* modifiy after here %%%%

        function setCurrentState(obj, x)
            obj.x = x;
        end

        function setCurrentObs(obj, y)
            obj.y = y;
        end

        function setReference(obj, rs)
            obj.rs = rs;
        end

        function setCurrentReward(obj, rew)
            obj.rew = rew;
        end

        function setCurrentStatus(obj, status)
            obj.status = status;
        end

        function setSamplingRate(obj, Ts)
            obj.Ts = Ts;
        end

        function setTmax(obj, Tmax)
            obj.Tmax = Tmax;
        end

        function setStateDimensions(obj, N)
            obj.state_dim = N;
        end

        function u = getCurrentControl(obj)
            u = obj.u;
        end

        function preallocate(obj)
            K = obj.max_n_episodes;
            N = size(0:obj.Ts:obj.Tmax,2);
            M = obj.state_dim;

            obj.x = nan(1,M);
            obj.xs = nan(N,M);
            obj.xss = nan(K,N,M);
    
            obj.y = nan(1);
            obj.ys = nan(N,1);
            obj.yss = nan(K,N,1);
            
            obj.r = nan(1);
            obj.rs = nan(N,1);
            obj.rss = nan(K,N,1);
            
            obj.u = nan(1);
            obj.us = nan(N-1,1);
            obj.uss = nan(K,N-1,1);
            
            obj.rew = nan(1);
            obj.rews = nan(N-1,1);
            obj.rewss = nan(K,N-1,1);
        end
    end
end

