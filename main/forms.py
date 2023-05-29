from django import forms

class AlfaForm(forms.Form):
    alfa = forms.FloatField(label='Введите значение alfa', initial=0.05, required=True)
