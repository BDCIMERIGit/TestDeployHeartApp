{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "249c42bb-4513-4ea6-9361-1547ae59dfc3",
   "metadata": {},
   "outputs": [
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Classification Report:\n",
      "               precision    recall  f1-score   support\n",
      "\n",
      "           0       0.86      0.86      0.86        29\n",
      "           1       0.88      0.88      0.88        32\n",
      "\n",
      "    accuracy                           0.87        61\n",
      "   macro avg       0.87      0.87      0.87        61\n",
      "weighted avg       0.87      0.87      0.87        61\n",
      "\n"
     ]
    },
    {
     "data": {
      "image/png": "iVBORw0KGgoAAAANSUhEUgAAAfUAAAGHCAYAAACposvbAAAAOXRFWHRTb2Z0d2FyZQBNYXRwbG90bGliIHZlcnNpb24zLjYuMywgaHR0cHM6Ly9tYXRwbG90bGliLm9yZy/P9b71AAAACXBIWXMAAA9hAAAPYQGoP6dpAABBX0lEQVR4nO3dd1gUV/828HtBXECKolI0iiI2rNghERBBJTYsgViiGFtEo8YaNBEsETGJxhJrROwlij5qDDasjyUYNRr1MUZBLBAjKggCInveP3zdXzaAwrIwy+z9yTXXBWfadzfgzTlzdkYhhBAgIiKiMs9I6gKIiIhINxjqREREMsFQJyIikgmGOhERkUww1ImIiGSCoU5ERCQTDHUiIiKZYKgTERHJBEOdiIhIJhjqVKZcvnwZQ4YMQe3atWFqagoLCwu0aNEC8+fPx+PHj0v03BcvXoSnpyesra2hUCjw3Xff6fwcCoUCYWFhOj/u20RFRUGhUEChUODYsWN51gsh4OzsDIVCAS8vL63OsWzZMkRFRRVpn2PHjhVYExHlVU7qAogKa/Xq1QgODkb9+vUxefJkuLi4ICcnB+fPn8eKFStw5swZ7Nq1q8TO//HHHyMjIwNbt25FpUqVUKtWLZ2f48yZM3jnnXd0ftzCsrS0xJo1a/IE9/Hjx3Hr1i1YWlpqfexly5ahSpUqCAoKKvQ+LVq0wJkzZ+Di4qL1eYkMCUOdyoQzZ85g1KhR8PX1xe7du6FUKtXrfH19MXHiRMTExJRoDb///juGDx8OPz+/EjtHu3btSuzYhREYGIhNmzbh+++/h5WVlbp9zZo1cHNzQ1paWqnUkZOTA4VCASsrK8nfE6KyhMPvVCbMnTsXCoUCq1at0gj018qXL48ePXqov1epVJg/fz4aNGgApVIJW1tbDBo0CPfu3dPYz8vLC40bN0ZcXBzat28Pc3NzODk5Yd68eVCpVAD+b2j65cuXWL58uXqYGgDCwsLUX//T630SEhLUbbGxsfDy8kLlypVhZmaGmjVrok+fPnj+/Ll6m/yG33///Xf07NkTlSpVgqmpKZo3b45169ZpbPN6mHrLli2YPn06qlWrBisrK/j4+ODGjRuFe5MB9OvXDwCwZcsWdVtqaip27tyJjz/+ON99Zs6cibZt28LGxgZWVlZo0aIF1qxZg38+K6pWrVq4evUqjh8/rn7/Xo90vK59w4YNmDhxIqpXrw6lUok///wzz/D7o0ePUKNGDbi7uyMnJ0d9/GvXrqFChQr46KOPCv1aieSIoU56Lzc3F7GxsWjZsiVq1KhRqH1GjRqFqVOnwtfXF3v27MHs2bMRExMDd3d3PHr0SGPb5ORkDBgwAAMHDsSePXvg5+eHkJAQbNy4EQDQtWtXnDlzBgDQt29fnDlzRv19YSUkJKBr164oX748IiMjERMTg3nz5qFChQp48eJFgfvduHED7u7uuHr1KhYvXozo6Gi4uLggKCgI8+fPz7P9tGnTcOfOHfzwww9YtWoVbt68ie7duyM3N7dQdVpZWaFv376IjIxUt23ZsgVGRkYIDAws8LWNHDkS27dvR3R0NHr37o1PP/0Us2fPVm+za9cuODk5wdXVVf3+/ftSSUhICBITE7FixQrs3bsXtra2ec5VpUoVbN26FXFxcZg6dSoA4Pnz5/jggw9Qs2ZNrFixolCvk0i2BJGeS05OFgDEhx9+WKjtr1+/LgCI4OBgjfZz584JAGLatGnqNk9PTwFAnDt3TmNbFxcX0blzZ402AGL06NEabaGhoSK/X6O1a9cKACI+Pl4IIcSOHTsEAHHp0qU31g5AhIaGqr//8MMPhVKpFImJiRrb+fn5CXNzc/H06VMhhBBHjx4VAMT777+vsd327dsFAHHmzJk3nvd1vXFxcepj/f7770IIIVq3bi2CgoKEEEI0atRIeHp6Fnic3NxckZOTI2bNmiUqV64sVCqVel1B+74+n4eHR4Hrjh49qtEeEREhAIhdu3aJwYMHCzMzM3H58uU3vkYiQ8CeOsnO0aNHASDPhKw2bdqgYcOGOHLkiEa7vb092rRpo9HWtGlT3LlzR2c1NW/eHOXLl8eIESOwbt063L59u1D7xcbGomPHjnlGKIKCgvD8+fM8Iwb/vAQBvHodAIr0Wjw9PVGnTh1ERkbiypUriIuLK3Do/XWNPj4+sLa2hrGxMUxMTDBjxgykpKTg4cOHhT5vnz59Cr3t5MmT0bVrV/Tr1w/r1q3DkiVL0KRJk0LvTyRXDHXSe1WqVIG5uTni4+MLtX1KSgoAwMHBIc+6atWqqde/Vrly5TzbKZVKZGZmalFt/urUqYPDhw/D1tYWo0ePRp06dVCnTh0sWrTojfulpKQU+Dper/+nf7+W1/MPivJaFAoFhgwZgo0bN2LFihWoV68e2rdvn++2v/zyCzp16gTg1acT/vvf/yIuLg7Tp08v8nnze51vqjEoKAhZWVmwt7fntXSi/4+hTnrP2NgYHTt2xK+//ppnolt+XgdbUlJSnnUPHjxAlSpVdFabqakpACA7O1uj/d/X7QGgffv22Lt3L1JTU3H27Fm4ublh/Pjx2Lp1a4HHr1y5coGvA4BOX8s/BQUF4dGjR1ixYgWGDBlS4HZbt26FiYkJ9u3bh4CAALi7u6NVq1ZanTO/CYcFSUpKwujRo9G8eXOkpKRg0qRJWp2TSG4Y6lQmhISEQAiB4cOH5zuxLCcnB3v37gUAeHt7A4B6ottrcXFxuH79Ojp27Kizul7P4L58+bJG++ta8mNsbIy2bdvi+++/BwBcuHChwG07duyI2NhYdYi/tn79epibm5fYx72qV6+OyZMno3v37hg8eHCB2ykUCpQrVw7GxsbqtszMTGzYsCHPtroa/cjNzUW/fv2gUCjw888/Izw8HEuWLEF0dHSxj01U1vFz6lQmuLm5Yfny5QgODkbLli0xatQoNGrUCDk5Obh48SJWrVqFxo0bo3v37qhfvz5GjBiBJUuWwMjICH5+fkhISMCXX36JGjVq4LPPPtNZXe+//z5sbGwwdOhQzJo1C+XKlUNUVBTu3r2rsd2KFSsQGxuLrl27ombNmsjKylLPMPfx8Snw+KGhodi3bx86dOiAGTNmwMbGBps2bcJPP/2E+fPnw9raWmev5d/mzZv31m26du2KBQsWoH///hgxYgRSUlLwzTff5PuxwyZNmmDr1q3Ytm0bnJycYGpqqtV18NDQUJw8eRIHDx6Evb09Jk6ciOPHj2Po0KFwdXVF7dq1i3xMIrlgqFOZMXz4cLRp0wYLFy5EREQEkpOTYWJignr16qF///4YM2aMetvly5ejTp06WLNmDb7//ntYW1ujS5cuCA8Pz/caurasrKwQExOD8ePHY+DAgahYsSKGDRsGPz8/DBs2TL1d8+bNcfDgQYSGhiI5ORkWFhZo3Lgx9uzZo74mnZ/69evj9OnTmDZtGkaPHo3MzEw0bNgQa9euLdKd2UqKt7c3IiMjERERge7du6N69eoYPnw4bG1tMXToUI1tZ86ciaSkJAwfPhzPnj2Do6Ojxuf4C+PQoUMIDw/Hl19+qTHiEhUVBVdXVwQGBuLUqVMoX768Ll4eUZmjEOIfd4ggIiKiMovX1ImIiGSCoU5ERCQTDHUiIiKZYKgTERHJBEOdiIhIJhjqREREMsFQJyIikglZ3nzGrNcPUpdAVOKe/Djs7RsRlXGmJZxSZq5j3r5RATIvLtVhJbohy1AnIiIqFIW8BqwZ6kREZLiK8HTAsoChTkREhktmPXV5vRoiIiIDxp46EREZLg6/ExERyYTMht8Z6kREZLjYUyciIpIJ9tSJiIhkQmY9dXn9iUJERGTA2FMnIiLDxeF3IiIimZDZ8DtDnYiIDBd76kRERDLBnjoREZFMyKynLq9XQ0REZMDYUyciIsMls546Q52IiAyXEa+pExERyQN76kRERDLB2e9EREQyIbOeurxeDRERkQFjT52IiAwXh9+JiIhkQmbD7wx1IiIyXOypExERyQR76kRERDIhs566vP5EISIiMmDsqRMRkeHi8DsREZFMyGz4naFORESGiz11IiIimWCoExERyYTMht/l9ScKERGRHgoPD0fr1q1haWkJW1tb+Pv748aNGxrbBAUFQaFQaCzt2rUr0nkY6kREZLgURtovRXD8+HGMHj0aZ8+exaFDh/Dy5Ut06tQJGRkZGtt16dIFSUlJ6mX//v1FOg+H34mIyHCV0vB7TEyMxvdr166Fra0tfv31V3h4eKjblUol7O3ttT4Pe+pERGS4itFTz87ORlpamsaSnZ1dqNOmpqYCAGxsbDTajx07BltbW9SrVw/Dhw/Hw4cPi/RyGOpERGS4FAqtl/DwcFhbW2ss4eHhbz2lEAITJkzAe++9h8aNG6vb/fz8sGnTJsTGxuLbb79FXFwcvL29C/2HAgAohBBCqzeiBGRlZcHU1LTYxzHr9YMOqiHSb09+HCZ1CUQlzrSELxKb94nUet8nmwfkCVylUgmlUvnG/UaPHo2ffvoJp06dwjvvvFPgdklJSXB0dMTWrVvRu3fvQtUkeU9dpVJh9uzZqF69OiwsLHD79m0AwJdffok1a9ZIXB0REVH+lEolrKysNJa3Bfqnn36KPXv24OjRo28MdABwcHCAo6Mjbt68WeiaJA/1OXPmICoqCvPnz0f58uXV7U2aNMEPP7DHTUREJeffHyErylIUQgiMGTMG0dHRiI2NRe3atd+6T0pKCu7evQsHB4dCn0fyUF+/fj1WrVqFAQMGwNjYWN3etGlT/O9//5OwMiIikj1FMZYiGD16NDZu3IjNmzfD0tISycnJSE5ORmZmJgAgPT0dkyZNwpkzZ5CQkIBjx46he/fuqFKlCnr16lXo80j+kbb79+/D2dk5T7tKpUJOTo4EFRERkaEoao9bW8uXLwcAeHl5abSvXbsWQUFBMDY2xpUrV7B+/Xo8ffoUDg4O6NChA7Zt2wZLS8tCn0fyUG/UqBFOnjwJR0dHjfYff/wRrq6uElVFRESGoLRC/W1z0s3MzHDgwIFin0fyUA8NDcVHH32E+/fvQ6VSITo6Gjdu3MD69euxb98+qcsjIiIZK61QLy2SX1Pv3r07tm3bhv3790OhUGDGjBm4fv069u7dC19fX6nLIyIiKjMk76kDQOfOndG5c2epyyAiIgPDnrqO3b17F/fu3VN//8svv2D8+PFYtWqVhFUREZFBKKXZ76VF8lDv378/jh49CgBITk6Gj48PfvnlF0ybNg2zZs2SuDoiIpKz0vqcemmRPNR///13tGnTBgCwfft2NGnSBKdPn8bmzZsRFRUlbXFERCRrcgt1ya+p5+TkqG+rd/jwYfTo0QMA0KBBAyQlJUlZGhERyZy+hrO2JO+pN2rUCCtWrMDJkydx6NAhdOnSBQDw4MEDVK5cWeLqiIiIyg7JQz0iIgIrV66El5cX+vXrh2bNmgEA9uzZox6WJyIiKgkcftcxLy8vPHr0CGlpaahUqZK6fcSIETA3N5ewMiIikj39zGatSR7qAGBsbKwR6ABQq1YtaYohIiKDoa89bm3pRajv2LED27dvR2JiIl68eKGx7sKFCxJVRUREcie3UJf8mvrixYsxZMgQ2Nra4uLFi2jTpg0qV66M27dvw8/PT+ryiIhIxuR2TV3yUF+2bBlWrVqFpUuXonz58pgyZQoOHTqEsWPHIjU1VeryiIiIygzJQz0xMRHu7u4AXj167tmzZwCAjz76CFu2bJGyNCIikjveJla37O3tkZKSAgBwdHTE2bNnAQDx8fFvff4sERFRcXD4Xce8vb2xd+9eAMDQoUPx2WefwdfXF4GBgejVq5fE1RERkZzJLdQln/2+atUqqFQqAMAnn3wCGxsbnDp1Ct27d8cnn3wicXVERCRn+hrO2pI81I2MjGBk9H8DBgEBAQgICJCwIiIiMhRyC3XJh98B4OTJkxg4cCDc3Nxw//59AMCGDRtw6tQpiSsjIiIqOyQP9Z07d6Jz584wMzPDxYsXkZ2dDQB49uwZ5s6dK3F1REQka5z9rltz5szBihUrsHr1apiYmKjb3d3deTc5IiIqUZwop2M3btyAh4dHnnYrKys8ffq09AsiIiKDoa/hrC3Je+oODg74888/87SfOnUKTk5OElRERESGQm49dclDfeTIkRg3bhzOnTsHhUKBBw8eYNOmTZg0aRKCg4OlLo+IiKjMkHz4fcqUKUhNTUWHDh2QlZUFDw8PKJVKTJo0CWPGjJG6PCIikjP97HBrTfJQB4CvvvoK06dPx7Vr16BSqeDi4gILCwupy6J/mNS7Gfzb1UK9d6yR+SIX5/73F6avj8PNB//30J1Vn3rgI+96Gvv9cuMhPD/fU9rlEpWYNatXYvF3CzBg4CBMCZkudTlUTPo6jK4tvQh1ADA3N0erVq2QlpaGw4cPo379+mjYsKHUZdH/176RPVb8fA2//vk3yhkbIWxAK+wL7QLXsTvxPPulersDF+5i5JIT6u9fvFRJUS5Rifj9ymXs+HEb6tWrL3UppCNyC3XJr6kHBARg6dKlAIDMzEy0bt0aAQEBaNq0KXbu3ClxdfRaz9kHsPHoTVy/+xRXEh5j5JITqGlrCdc6VTS2e5GTi7+eZqqXJ+nZElVMpFvPMzIQMnUyQmfOgZW1tdTlkI5wopyOnThxAu3btwcA7Nq1CyqVCk+fPsXixYsxZ84ciaujgliZlweAPKHdvrED7kQNwOXvP8D3we+hqrWpFOUR6dzcObPg4eGJdm7uUpdCOsRQ17HU1FTY2NgAAGJiYtCnTx+Ym5uja9euuHnzpsTVUUEihrTFf68l41riE3XbwQv3MGThMfjN2I/P155DS+eq+HnW+yhfTvIfM6Ji+Xn/T7h+/RrGfjZR6lKI3kjya+o1atTAmTNnYGNjg5iYGGzduhUA8OTJE5iavr2Xl52drb617GsiNwcKY5MC9qDiWjjCHU1q2aDjtL0a7Tv+e1v99bXEJ7hw62/cWPkh/FrVxH/OJpRylUS6kZyUhPnzvsKKVZFQKpVSl0O6pp8dbq1JHurjx4/HgAEDYGFhAUdHR3h5eQF4NSzfpEmTt+4fHh6OmTNnarQZ1+8Ok4Y9SqJcg7dgmBu6ta4Jn+n7cD/l+Ru3TX6SicS/0+HsYFVK1RHp3rVrV/E4JQX9Anqr23Jzc/Hr+Ths3bIJcRevwNjYWMIKqTj0dRhdW5KHenBwMNq0aYO7d+/C19dX/RhWJyenQl1TDwkJwYQJEzTabAduKpFaDd3C4W7o0bYWOn35E+48TH/r9jaWSrxTpQKSnrw5/In0Wdt27bBjt+aoVOj0ENRycsKQocMZ6GUcQ70EtGrVCq1atdJo69q1a6H2VSqVeYbEOPSue9+NcEegRx18EH4I6Zk5sKtoBgBIff4CWS9yUcG0HL4IbIHdZxOQ9Pg5HG0tMWtgK6SkZWPP2TsSV0+kvQoVLFC3rub9F8zMzVHRumKedip7ZJbp0oT6hAkTMHv2bFSoUCFPL/vfFixYUEpV0ZuM9HMBABya002jffji49h49CZyVQKNHG3Qv0NdVDQvj+Qnz3H89yR89E0s0rNypCiZiOit2FPXgYsXLyInJ0f9dUHk9maXZWa9fnjj+qwXuegxK6aUqiGS1pqoDVKXQJQvSUL96NGj+X5NRERUmuTWd9SLa+pCCKSkpEChUKBy5cpSl0NERAZCbiPCkt4VJDk5GYMGDUKlSpVgZ2cHW1tbVKpUCR9//DH++usvKUsjIiIDoFBov+gjyXrqaWlpcHd3R3p6OoYMGYIGDRpACIFr165hy5YtOHXqFC5cuMCntRERUYkxMtLTdNaSZKG+aNEiGBsb4+rVq6hatarGui+++ALvvvsuFi9ejGnTpklUIRERyZ2+9ri1Jdnw+08//YRp06blCXQAsLW1RUhICPbu3ZvPnkRERJQfyUL9jz/+gLt7wU87cnd3x40bN0qxIiIiMjRye0qbpNfUK1asWOD6ihUrIi0trfQKIiIig6On2aw1yUJdCKG+z3t+FAoFhBClWBERERkafe1xa0vSUK9Xr16BbygDnYiIShpDXUfWrl0r1amJiIgAcPhdZwYPHizVqYmIiGRJL24TS0REJAUOvxMREcmEzDKdoU5ERIaLPXUiIiKZkFmm61eov/4Ym9z+ciIiIv0kt7yR9NGrr61fvx5NmjSBmZkZzMzM0LRpU2zYsEHqsoiIiHQiPDwcrVu3hqWlJWxtbeHv75/nVuhCCISFhaFatWowMzODl5cXrl69WqTzSB7qCxYswKhRo/D+++9j+/bt2LZtG7p06YJPPvkECxculLo8IiKSsdJ6nvrx48cxevRonD17FocOHcLLly/RqVMnZGRkqLeZP38+FixYgKVLlyIuLg729vbw9fXFs2fPCv96hMS3bqtduzZmzpyJQYMGabSvW7cOYWFhiI+PL/IxzXr9oKvyiPTWkx+HSV0CUYkzLeGLxG3Dj2u974kJ7ZCdna3RplQqoVQq37rv33//DVtbWxw/fhweHh4QQqBatWoYP348pk6dCgDIzs6GnZ0dIiIiMHLkyELVJHlPPSkpKd+ntbm7uyMpKUmCioiIyFAUp6ceHh4Oa2trjSU8PLxQ501NTQUA2NjYAADi4+ORnJyMTp06qbdRKpXw9PTE6dOnC/16JA91Z2dnbN++PU/7tm3bULduXQkqIiIiQ1GcR6+GhIQgNTVVYwkJCXnrOYUQmDBhAt577z00btwYAJCcnAwAsLOz09jWzs5Ova4wJJ/9PnPmTAQGBuLEiRN49913oVAocOrUKRw5ciTfsCciItKV4kx+L+xQ+7+NGTMGly9fxqlTp/KpR7MgIUSRZuhL3lPv06cPzp07hypVqmD37t2Ijo5GlSpV8Msvv6BXr15Sl0dERKQzn376Kfbs2YOjR4/inXfeUbfb29sDQJ5e+cOHD/P03t9E8p46ALRs2RIbN26UugwiIjIwpfU5dSEEPv30U+zatQvHjh1D7dq1NdbXrl0b9vb2OHToEFxdXQEAL168wPHjxxEREVHo8+hFqBMREUmhtO49M3r0aGzevBn/+c9/YGlpqe6RW1tbw8zMDAqFAuPHj8fcuXNRt25d1K1bF3PnzoW5uTn69+9f6PNIFupGRkZv/QtJoVDg5cuXpVQREREZmtLqqS9fvhwA4OXlpdG+du1aBAUFAQCmTJmCzMxMBAcH48mTJ2jbti0OHjwIS0vLQp9HslDftWtXgetOnz6NJUuWQOKP0BMRkcyV5vD72ygUCoSFhSEsLEzr80gW6j179szT9r///Q8hISHYu3cvBgwYgNmzZ0tQGRERGQqZ3fpd+tnvAPDgwQMMHz4cTZs2xcuXL3Hp0iWsW7cONWvWlLo0IiKiMkPSUE9NTcXUqVPh7OyMq1ev4siRI9i7d6/6w/hEREQlqTg3n9FHkg2/z58/HxEREbC3t8eWLVvyHY4nIiIqSXqazVqTLNQ///xzmJmZwdnZGevWrcO6devy3S46OrqUKyMiIkOhrz1ubUkW6oMGDZLdm0lERGWL3GJIslCPioqS6tREREQAACOZpbpezH4nIiKi4uNtYomIyGDJrKPOUCciIsMlt7ldDHUiIjJYRvLKdIY6EREZLvbUiYiIZEJmmc7Z70RERHLBnjoRERksBeTVVWeoExGRweJEOSIiIpngRDkiIiKZkFmmM9SJiMhw8d7vREREpJfYUyciIoMls446Q52IiAwXJ8oRERHJhMwynaFORESGS24T5RjqRERksOQV6YUM9T179hT6gD169NC6GCIiItJeoULd39+/UAdTKBTIzc0tTj1ERESlxiAnyqlUqpKug4iIqNTx3u9EREQyYZA99X/LyMjA8ePHkZiYiBcvXmisGzt2rE4KIyIiKmkyy/Sih/rFixfx/vvv4/nz58jIyICNjQ0ePXoEc3Nz2NraMtSJiKjMkFtPvcj3fv/ss8/QvXt3PH78GGZmZjh79izu3LmDli1b4ptvvimJGomIiKgQihzqly5dwsSJE2FsbAxjY2NkZ2ejRo0amD9/PqZNm1YSNRIREZUII4X2iz4qcqibmJiohyvs7OyQmJgIALC2tlZ/TUREVBYoFAqtF31U5Gvqrq6uOH/+POrVq4cOHTpgxowZePToETZs2IAmTZqURI1EREQlQj+jWXtF7qnPnTsXDg4OAIDZs2ejcuXKGDVqFB4+fIhVq1bpvEAiIqKSYqRQaL3ooyL31Fu1aqX+umrVqti/f79OCyIiIiLt8OYzRERksPS0w621Iod67dq13zhB4Pbt28UqiIiIqLTo64Q3bRU51MePH6/xfU5ODi5evIiYmBhMnjxZV3URERGVOJlletFDfdy4cfm2f//99zh//nyxCyIiIiot+jrhTVtFnv1eED8/P+zcuVNXhyMiIipxCoX2iz7SWajv2LEDNjY2ujocERERFZFWN5/558QCIQSSk5Px999/Y9myZTotjoiIqCQZ/ES5nj17arwJRkZGqFq1Kry8vNCgQQOdFqetJz8Ok7oEohJXqfUYqUsgKnGZF5eW6PF1NlytJ4oc6mFhYSVQBhERUemTW0+9yH+kGBsb4+HDh3naU1JSYGxsrJOiiIiISoPcntJW5J66ECLf9uzsbJQvX77YBREREZUWfQ1nbRU61BcvXgzg1VDFDz/8AAsLC/W63NxcnDhxQm+uqRMRERmiQof6woULAbzqqa9YsUJjqL18+fKoVasWVqxYofsKiYiISojcrqkXOtTj4+MBAB06dEB0dDQqVapUYkURERGVBoMd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\n",
      "text/plain": [
       "<Figure size 600x400 with 2 Axes>"
      ]
     },
     "metadata": {},
     "output_type": "display_data"
    },
    {
     "name": "stdout",
     "output_type": "stream",
     "text": [
      "Model saved as 'HeartModel.pkl'\n"
     ]
    }
   ],
   "source": [
    "#Test Deploy Heart Disease App Streamlit\n",
    "\n",
    "import pandas as pd\n",
    "import numpy as np\n",
    "import seaborn as sns\n",
    "import matplotlib.pyplot as plt\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.svm import SVC\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.metrics import classification_report, confusion_matrix\n",
    "import joblib\n",
    "\n",
    "# Load the dataset\n",
    "file_path = 'heart.csv'  # Update this if necessary\n",
    "df = pd.read_csv(file_path)\n",
    "\n",
    "# Split into features (X) and target (y)\n",
    "X = df.drop(columns=['target'])  # Assuming 'target' is the label column\n",
    "y = df['target']\n",
    "\n",
    "# Train-test split\n",
    "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Create pipeline with SVM model\n",
    "svm_pipeline = Pipeline([\n",
    "    ('scaler', StandardScaler()),\n",
    "    ('svc', SVC(C=1, gamma='scale', kernel='linear'))\n",
    "])\n",
    "\n",
    "# Train the model\n",
    "svm_pipeline.fit(X_train, y_train)\n",
    "\n",
    "# Make predictions\n",
    "y_pred = svm_pipeline.predict(X_test)\n",
    "\n",
    "# Print classification report\n",
    "print(\"Classification Report:\\n\", classification_report(y_test, y_pred))\n",
    "\n",
    "# Generate confusion matrix\n",
    "cm = confusion_matrix(y_test, y_pred)\n",
    "plt.figure(figsize=(6,4))\n",
    "sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No Disease', 'Disease'], yticklabels=['No Disease', 'Disease'])\n",
    "plt.xlabel('Predicted')\n",
    "plt.ylabel('Actual')\n",
    "plt.title('Confusion Matrix')\n",
    "plt.show()\n",
    "\n",
    "# Save the model\n",
    "joblib.dump(svm_pipeline, 'HeartModel.pkl')\n",
    "print(\"Model saved as 'HeartModel.pkl'\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "id": "fa088f0e-1339-4750-812f-e34d61e95191",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.10.8"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
