import {Point, Vector, Matrix} from '@mathigon/fermat';
import {CoordinateSystem, Step} from '../shared/types';
import '../shared/shared';


function sigmoid(a: number, b: number) {
  return (x: number) => 1 / (1 + Math.exp(-(a + b * x)));
}

function quadratic(β_0: number, β_1: number, β_2: number) {
  return (x: number) => β_0+ β_1*x+ β_2*Math.pow(x, 2);
}

export function linearExercise($step: Step) {
  const $chart = $step.$('x-coordinate-system') as CoordinateSystem;
  const points = [new Point(1,1), new Point(2, 4), new Point(3, 8), new Point(5, 23),
    new Point(4,15), new Point(3.5, 12)];

  $step.model.watch((s: any) => {
    const fn = quadratic(s.β_0, s.β_1, s.β_2);
    s.$step.model.loss = points.map(x => Math.pow((x.y - fn(x.x)), 2))
    .reduce((a, b) => a + b, 0);
    
    $chart.setFunctions(fn);
    $chart.drawPoints(points);
  });
}

export function qdaExercise($step: Step) {
  const $chart = $step.$('x-coordinate-system') as CoordinateSystem;
  const points = [new Point(1,1), new Point(2, 2), new Point(2, 4), new Point(2.8, 0),
    new Point(2.5,2), new Point(0, 8), new Point(0, 4), new Point(0, 2), new Point(0, 8), 
    new Point(1, 6), new Point(1, 2)];

  $step.model.watch((s: any) => {
    const fn = quadratic(s.β_0, s.β_1, s.β_2);
    $chart.setFunctions(fn);
    $chart.drawPoints(points);
  });
}

export function logisticAnimationExercise($step: Step) {
  const $chart = $step.$('x-coordinate-system') as CoordinateSystem;
  const zeros = [-1.2, -0.8, -0.7, 0.4, -2.4, 1.13];
  const ones = [2.2, 1.3, 0.8, 2.5, 2.62];
  const points0 = zeros.map(p => new Point(p, 0));
  const points1 = ones.map(p => new Point(p, 1));

  $step.model.watch((s: any) => {
    const fn = sigmoid(s.α, s.β);
    var onesloss = ones.map(x => Math.log(1 / fn(x)))
        .reduce((a, b) => a + b, 0);
    var zerosloss = zeros.map(x => Math.log(1 / (1 - fn(x))))
        .reduce((a, b) => a + b, 0);
    s.$step.model.loss = Math.round(10 ** 3 * (onesloss + zerosloss)) / 10 ** 3;
    
    $chart.setFunctions(fn);
    $chart.drawPoints(points0);
    $chart.drawPoints(points1);
  });
}

function line(wi: number, wj: number, b: number, lambda:number) {
  return (x: number) => (b-(wi*x))/wj;
}

function line1(wi: number, wj: number, b: number, lambda:number) {
  return (x: number) => (1+b-(wi*x))/wj;
}

function line2(wi: number, wj: number, b: number, lambda:number) {
  return (x: number) => (-1+b-(wi*x))/wj;
}

export function svmAnimationExercise($step: Step) {
  const $chart = $step.$('x-coordinate-system') as CoordinateSystem;
  const group1 = [new Point(2.5,6), new Point(3, 6), new Point(3.5,5), new Point(4.5, 7),
      new Point(4,9), new Point(3.5, 6), new Point(3.5,8), new Point(2.5, 6)];
  const group2 = [new Point(4,3), new Point(4, 2), new Point(6,2), new Point(6, 4),
      new Point(6,1), new Point(4, 1), new Point(4.5,2), new Point(5, 3)];

   $step.model.watch((s: any) => {
     const fn = line(s.wi, s.wj, s.b, s.lambda);
     const fn1 = line1(s.wi, s.wj, s.b, s.lambda);
     const fn2 = line2(s.wi, s.wj, s.b, s.lambda);
     var group1loss = group1.map(x => Math.max(0, 1-1*(s.wi*x.x+s.wj*x.y-s.b)))
     .reduce((a, b) => a + b, 0);
     var group2loss = group2.map(x => Math.max(0, 1+1*(s.wi*x.x+s.wj*x.y-s.b)))
     .reduce((a, b) => a + b, 0);
     var lambdaloss = s.wi*s.wi+s.wj*s.wj;
     s.$step.model.loss = ((1/(group1.length+group2.length))*(group1loss+group2loss)) + 
     s.lambda*lambdaloss;
    
    $chart.setFunctions(fn, fn1, fn2);
    $chart.drawPoints(group1);
    $chart.drawPoints(group2);
  });
}

 function cdf(mu: number, sigma: number, to: number) {
    var z = (to-mu)/Math.sqrt(2*sigma*sigma);
    var t = 1/(1+0.3275911*Math.abs(z));
    var a1 =  0.254829592;
    var a2 = -0.284496736;
    var a3 =  1.421413741;
    var a4 = -1.453152027;
    var a5 =  1.061405429;
    var erf = 1-(((((a5*t + a4)*t) + a3)*t + a2)*t + a1)*t*Math.exp(-z*z);
    var sign = 1;
    if(z < 0)
    {
        sign = -1;
    }
    return (1/2)*(1+sign*erf);
}

function dr(wi: number, wj: number, b: number, lambda:number) {
  return (x: number) => (-1+b-(wi*x))/wj;
}

export function likelihoodRatioExercise($step: Step) {
   const $chart = $step.$('x-coordinate-system') as CoordinateSystem;
  
   var pointArray: Point[] = [];
    $step.model.watch((s: any) => {
        var far = 1-cdf(0, 1, (Math.log(s.t)/s.mu)+s.mu/2)
        var dr = 1-cdf(s.mu, 1, (Math.log(s.t)/s.mu)+s.mu)
        var point = new Point(far, dr);
        pointArray.push(point);
      $chart.drawPoints(pointArray); 
   });
  }

function normal(a: Point, b: Point, sigma: number){
  return Math.exp(-1* Math.pow(Point.distance(a, b), 2)/(2*Math.pow(sigma, 2)));
}

export function dimensionReductionExercise($step: Step) {
  const $chart = $step.$('x-coordinate-system') as CoordinateSystem;
  const group = [new Point(0, 0), new Point(0, 1), new Point(1, 1), new Point(4, 0)]
  
    $step.model.watch((s: any) => {
      var numerator = normal(group[1], group[0], s.sigma);
      var sum  = normal(group[1], group[0],s.sigma)+ normal(group[2], group[0],s.sigma)+ 
      normal(group[3], group[0],s.sigma);
      var value = numerator/sum;
    s.$step.model.value = value;
    $chart.drawPoints(group);
  });
}
