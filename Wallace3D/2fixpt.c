#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define MAX_DECIMAL_BIT   2
#define MAX_FLOAT_BIT     2

int main()
{
  int _i, _j, sign;
  float x, ret, _k;
  while(scanf("%f", &x) != EOF)
  {
    //sheep add
    if(x < 0) sign = -1;
    else sign = 1;
    x += powf(2, - (MAX_FLOAT_BIT + 1)) * sign;         //先進位
    if(abs(x) > ((1 << (MAX_DECIMAL_BIT - 1)) - 1))     //如果加完超過整數能表示的就先不要進位
      x -= powf(2, -(MAX_FLOAT_BIT + 1)) * sign;
    
    ret = 0;

    // reserve a sign bit
    ret += (int) abs(x) > ((1 << (MAX_DECIMAL_BIT - 1)) - 1) ? ((1 << (MAX_DECIMAL_BIT - 1)) - 1) * sign: (int) abs(x) * sign;

    ///我覺得如果x已經大於整數能表示的，小數bit部分就直接全部填一。
    ///待修改。     sheep留

    for(_i = 0, _j = -1, _k = 2 * (x - (int) x); _i < MAX_FLOAT_BIT - 1; _i++, _j--, _k+=_k)
    {
      ret += abs(_k) - 1 >= 0 ? powf(2, _j) * sign: 0;  //change > to >= and add mult sign
      printf("[%d] : %f\n", _i, _k);
      _k -= (int)_k;
    }
    //is j - 1 not j + 1
    printf("j is %d, k is %f\n", _j, _k);
    printf("ret is %f\n", ret);
    ret += abs(_k) - 1 > 0 ? powf(2, _j) * sign : (abs(_k) * 2 - 1 > 0 ? powf(2, _j) * sign : 0);
    //ret += abs(_k) - 1 > 0 ? powf(2, _j) * sign : (abs(_k) - powf(2, _j - 1)> 0 ? powf(2, _j) * sign : 0);
    printf("%f after fixed point conversion: %f\n", x, ret);
  }
  return 0;
}
