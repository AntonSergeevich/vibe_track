# core/forms.py
"""Формы кабинета."""
from django.contrib.auth.forms import UserCreationForm

from .models import User


class RegisterForm(UserCreationForm):
    """Регистрация под нашу модель пользователя.

    Стандартная UserCreationForm жёстко привязана к auth.User, а у нас своя
    модель — без этой формы регистрация падает на проверке имени.
    """

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email")
